"""
batch_pipeline.py — Queue-based pipeline wrapper for your multi-camera
detection → tracking/ID workflow, tuned for stability & clean EOF drain:
- Reader→Detect is lossless (blocking put) to prevent pre-detect drops
- Detect→Track is lossless (blocking put) to prevent writer drops
- STOP delivery is non-blocking & STOP-safe (never dropped)
- Reader flushes pending items on EOF to speed pipeline drain
- Main loop waits for detect/track to finish naturally after EOF
- _safe_shutdown is idempotent and avoids HighGUI teardown (track thread closes windows)
- Periodic health logs; optional psutil RSS counter if installed
"""

import cv2
import time
import threading
import itertools
import queue
from typing import List, Any

# Optional memory usage (if psutil installed)
try:
    import psutil, os
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PROC = None

class _Stop: pass
STOP = _Stop()

# -------- Helpers: STOP-safe, lossless writer path, and (optional) drop-oldest --------
import queue as _q

def q_put_stop(q):
    """
    Deliver STOP without blocking.
    - Try put_nowait.
    - If full, drop exactly one NON-STOP item to make room.
    - Never drop an existing STOP. If queue is already saturated with STOPs, do nothing.
    """
    if q is None:
        return
    try:
        q.put_nowait(STOP)
        return
    except _q.Full:
        pass

    popped_stops = 0
    freed = False
    while True:
        try:
            oldest = q.get_nowait()
            if oldest is STOP:
                popped_stops += 1
                continue
            else:
                freed = True
                break
        except _q.Empty:
            break

    # Requeue any STOPs we popped
    for _ in range(popped_stops):
        try:
            q.put_nowait(STOP)
        except _q.Full:
            break

    if freed:
        try:
            q.put_nowait(STOP)
        except _q.Full:
            pass

def q_put_lossless(q, item, timeout=1.0):
    """For detect→track (writer) path: don't drop; block briefly."""
    q.put(item, timeout=timeout)

def q_put_drop_oldest(q: "queue.Queue", item):
    """Optional helper if you ever want realtime reader→detect dropping."""
    try:
        q.put_nowait(item)
        return
    except _q.Full:
        pass
    try:
        q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(item)
    except Exception:
        pass

def q_get_with_timeout(q: "queue.Queue", timeout=1.0):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


class PipelineMixin:
    # =============== Public Entry ===============
    def batch_frames_queue(self, data, file_name, is_show_image, cap=None, tracking_cap=None,
                           isFile=False, isCam=False):
        # Open readers
        video_readers = [cv2.VideoCapture(vdo) for vdo in data]

        self._user_quit = False      # only True when 'q' pressed or KeyboardInterrupt
        self._eof_flag  = False      # True when reader hits EOF on any camera
        # Save flags
        self._is_save_video = cap is not None
        self._is_save_tracking_video = tracking_cap is not None
        self._cap = cap
        self._tracking_cap = tracking_cap
        self._show = bool(is_show_image)

        # Queues and control
        self._q_read   = queue.Queue(maxsize=10)
        self._q_detect = queue.Queue(maxsize=10)
        self._stop_ev  = threading.Event()
        self._batch_id = itertools.count()

        self._frame_skip = getattr(self, "frame_skip_count", 3)
        self._resolution = getattr(self, "resolution", (1024, 768))
        if not hasattr(self, "camera_heights"):
            self.camera_heights = []

        print('are we saving video? ', self._is_save_video)
        print('are we saving tracking video? ', self._is_save_tracking_video)
        print(f'Frame skip: {self._frame_skip}, Resolution: {self._resolution}')
        print(f'current values small size {getattr(self, "SMALL_SIZE", None)}')

        # Launch workers (non-daemon so we can join cleanly)
        t_reader = threading.Thread(target=self._reader_worker, args=(video_readers,))
        t_detect = threading.Thread(target=self._detect_worker)
        t_track  = threading.Thread(target=self._trackid_worker)

        t_reader.start(); t_detect.start(); t_track.start()

        last_log = time.time()
        try:
            # Drain-aware wait loop
            while True:
                if self._stop_ev.is_set() or getattr(self, "_user_quit", False):
                    break

                r_alive = t_reader.is_alive()
                d_alive = t_detect.is_alive()
                tr_alive = t_track.is_alive()

                # Reader finished via EOF: let detect & track drain and exit on STOP
                if not r_alive and self._eof_flag:
                    if not d_alive and not tr_alive:
                        break  # all done
                    # periodic health log
                    now = time.time()
                    if now - last_log >= 15:
                        q_read_sz   = self._q_read.qsize() if self._q_read else 0
                        q_detect_sz = self._q_detect.qsize() if self._q_detect else 0
                        mem = ""
                        if _PROC is not None:
                            try:
                                rss = _PROC.memory_info().rss / (1024*1024)
                                mem = f" | RSS={rss:.1f}MB"
                            except Exception:
                                mem = ""
                        print(f"[health] readQ={q_read_sz} detectQ={q_detect_sz}{mem}")
                        last_log = now
                    time.sleep(0.05)
                    continue

                # If any worker died for other reasons, break and go to finally
                if not r_alive or not d_alive or not tr_alive:
                    break

                # normal health log
                now = time.time()
                if now - last_log >= 15:
                    q_read_sz   = self._q_read.qsize()
                    q_detect_sz = self._q_detect.qsize()
                    mem = ""
                    if _PROC is not None:
                        try:
                            rss = _PROC.memory_info().rss / (1024*1024)
                            mem = f" | RSS={rss:.1f}MB"
                        except Exception:
                            mem = ""
                    print(f"[health] readQ={q_read_sz} detectQ={q_detect_sz}{mem}")
                    last_log = now
                time.sleep(0.1)

            print("Thread states → reader:", t_reader.is_alive(),
                  "detect:", t_detect.is_alive(),
                  "track:", t_track.is_alive())
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down...")
            self._safe_shutdown()
        finally:
            # Only force-stop if it wasn't a normal EOF drain
            #if not getattr(self, "_eof_flag", False):
            #    self._safe_shutdown()

            for t in (t_reader, t_detect, t_track):
                try: t.join(timeout=2.0)
                except: pass
            # release readers
            for cap_ in video_readers:
                try: cap_.release()
                except: pass

            if getattr(self, "_user_quit", False):
                return -1
            if getattr(self, "_eof_flag", False):
                return 1
            return 0

    # =============== Workers ===============
    def _reader_worker(self, video_readers: List[cv2.VideoCapture]):
        """
        Per-cam ready buffers:
        - Each cam produces a kept frame only on skip boundary
        - When all cams have a kept frame, emit a batch
        - Reader→detect is LOSSLESS (blocking put with short timeout)
        - On EOF, flush pending items and then send STOP
        """
        n = len(video_readers)
        frame_counters = [0] * n
        ready = [None] * n
        last_log = time.time()

        # For gentle backpressure
        maxsz = getattr(self._q_read, "maxsize", 0)
        high_water = 0.8 * maxsz if maxsz else None

        while not self._stop_ev.is_set():
            progressed = False

            # Read/keep per cam
            for i, cap in enumerate(video_readers):
                if ready[i] is not None:
                    continue

                ok, frame = cap.read()
                if not ok:
                    self._eof_flag = True
                    print("[reader] EOF reached — flushing readQ and signaling STOP")
                    # Flush any pending pre-detect items so STOP is next
                    try:
                        while True:
                            _ = self._q_read.get_nowait()
                    except Exception:
                        pass
                    q_put_stop(self._q_read)  # STOP MUST be delivered
                    return
                frame = self.apply_clahe_ycrcb(frame, clip=2.5, tile=8)  #B
                frame = self.unsharp_mask(frame, radius=1.4, amount=1.0)  #B
                frame_counters[i] += 1
                if frame_counters[i] % (self._frame_skip + 1) != 0:
                    progressed = True
                    continue

                try:
                    frame = cv2.resize(frame, self._resolution, interpolation=cv2.INTER_AREA)
                except Exception as e:
                    print("Resize error:", e)
                    q_put_stop(self._q_read)
                    return

                ready[i] = frame
                progressed = True

            # Emit when all cams ready
            if all(r is not None for r in ready):
                images = ready

                if not self.camera_heights:
                    self.camera_heights = [im.shape[0] for im in images]

                bid = next(self._batch_id)

                # Backpressure: if queue is near full, pause reader a bit
                if high_water is not None:
                    while self._q_read.qsize() > high_water and not self._stop_ev.is_set():
                        time.sleep(0.005)

                # LOSSLESS handoff to detector (blocks briefly instead of dropping)
                try:
                    self._q_read.put((bid, images, {"camera_heights": self.camera_heights}), timeout=1.0)
                except queue.Full:
                    # If we still couldn't enqueue, initiate STOP to avoid deadlock
                    print("[reader] read queue full; sending STOP")
                    q_put_stop(self._q_read)
                    return

                ready = [None] * n

                if time.time() - last_log >= 10:
                    print(f"[reader] emitted bid={bid} qsize={self._q_read.qsize()}")
                    last_log = time.time()

            if not progressed:
                time.sleep(0.001)

    def _detect_worker(self):
        last_log = time.time()
        while not self._stop_ev.is_set():
            item = q_get_with_timeout(self._q_read, timeout=1.0)
            if item is None:
                # If EOF and nothing left to read, forward STOP and exit
                if getattr(self, "_eof_flag", False) and (self._q_read is None or self._q_read.empty()):
                    print("[detect] EOF observed & readQ empty — forwarding STOP")
                    q_put_stop(self._q_detect)
                    return
                continue
            if item is STOP:
                print("[detect] STOP received — forwarding to track and exiting")
                q_put_stop(self._q_detect)
                return

            bid, images, meta = item

            # Backpressure: if track queue is ~80% full, pause detection briefly
            try:
                maxsz = self._q_detect.maxsize or 0
            except Exception:
                maxsz = 0
            if maxsz:
                while self._q_detect.qsize() > 0.8 * maxsz and not self._stop_ev.is_set():
                    time.sleep(0.01)

            # Run detector
            try:
                detections = list(self.stupid_detector(images))
            except Exception as e:
                print("Error in detection:", e)
                q_put_stop(self._q_detect)
                return

            # LOSSLESS handoff to writer/track
            try:
                q_put_lossless(self._q_detect, (bid, images, detections, meta), timeout=1.0)
            except Exception:
                print("[detect] could not deliver to track in time; initiating STOP")
                q_put_stop(self._q_detect)
                return

            if time.time() - last_log >= 10:
                print(f"[detect] bid={bid} outQ={self._q_detect.qsize()}")
                last_log = time.time()

    def _trackid_worker(self):
        last_log = time.time()
        while not self._stop_ev.is_set():
            item = q_get_with_timeout(self._q_detect, timeout=1.0)
            if item is None:
                continue
            if item is STOP:
                # UI cleanup
                print("[track] STOP received — exiting")
                if self._show:
                    try: cv2.destroyAllWindows()
                    except: pass
                return

            bid, images, detections, meta = item

            # === Your per-cam CoreProcess stage ===
            frame_infos = []
            for cam_idx, (frame, outputs) in enumerate(zip(images, detections)):
                instances = outputs["instances"].to("cpu")
                h, w = frame.shape[:2]
                boxes, masks_np = self.sortBoxAndMask(
                    instances.pred_boxes.tensor.numpy(),
                    instances.pred_masks.numpy(),
                    is_last_cam=(cam_idx == 1),  # hardcoded for 4-5-6 pair
                    y2_threshold=h - 40
                )
                info = self.CoreProcess(boxes, masks_np, h, w, frame, cam_idx)
                frame_infos.append(info)
                images[cam_idx] = frame  # modify in-place as needed

            # === Cross-cam merge, draw, and output ===
            _ = self._merge_draw_and_write(images, frame_infos, meta)

            if time.time() - last_log >= 10:
                print(f"[track] bid={bid}")
                last_log = time.time()

    # =============== Drawing / Merge kept from your code ===============
    def _merge_draw_and_write(self, images, all_frame_infos, meta):
        # 1) Duplicate handling across cameras
        all_tracking_ids = [item for sublist in all_frame_infos for item in sublist.tracked_ids]
        all_duplicate_indexes = self.get_duplicate_indexes(
            [item for sublist in all_frame_infos for item in sublist.predicted_ids]
        )
        tracking_to_merge = self.get_Tracking_To_Merge(all_tracking_ids)

        all_duplicate_tracking_ids = []
        for duplicate in all_duplicate_indexes:
            if duplicate in ("Identifying", "Reidentifying", "unknown"):
                continue
            duplicate_tracking_ids = [all_tracking_ids[i] for i in all_duplicate_indexes[duplicate]]
            is_same = all(val == duplicate_tracking_ids[0] for val in duplicate_tracking_ids)
            if not is_same:
                for tid in duplicate_tracking_ids:
                    all_duplicate_tracking_ids.append(tid)

        if len(all_duplicate_tracking_ids) > 0:
            try:
                self.reset_duplicate_tracking_identification(all_duplicate_tracking_ids)
            except Exception as e:
                print("Warning: reset_duplicate_tracking_identification failed:", e)

        # 2) Build merged boxes across cameras
        from collections import defaultdict
        boxes_to_merge = defaultdict(list)
        for frame_info in all_frame_infos:
            mask_count = 0
            images[frame_info.cam_counter] = cv2.addWeighted(
                frame_info.colored_mask, 0.3,
                images[frame_info.cam_counter], 1 - 0.3, 0
            )
            for index in frame_info.tracked_indexes:
                x1, y1, x2, y2, area = map(int, frame_info.boxes[index])
                tracking_id = frame_info.tracked_ids[mask_count]
                label = 'Identifying' if tracking_id in all_duplicate_tracking_ids else f"{str(frame_info.predicted_ids[mask_count])}"
                if tracking_id in tracking_to_merge.keys():
                    sum_y = 0
                    if frame_info.cam_counter > 0:
                        sum_y = sum(meta.get("camera_heights", [])[: frame_info.cam_counter])
                    boxes_to_merge[tracking_id] += [[x1, y1+sum_y, x2, y2+sum_y], label]
                else:
                    try:
                        self.draw_bounding_box(
                            images[frame_info.cam_counter], (x1,y1,x2,y2), label,
                            str(frame_info.tracked_ids[mask_count]), font_scale=1
                        )
                    except Exception as e:
                        print("draw_bounding_box error:", e)
                mask_count += 1

        # 3) Update missed counts
        self.IncreaseMissedCount(all_tracking_ids)

        # 4) Compose final stacked image
        stacked_image = self.stack_image_from_bottom_to_top(images)
        if stacked_image is None:
            print("Stacked image is None")
            return None

        # 5) Draw merged boxes on composed image
        for key, box_to_merge in boxes_to_merge.items():
            try:
                if len(box_to_merge) >= 4:
                    x1,y1,x2,y2 = self.combine_boxes(box_to_merge[0], box_to_merge[2])
                    label = box_to_merge[1]
                    if y2 - y1 > 650:
                        x1,y1,x2,y2 = box_to_merge[0]
                        self.draw_bounding_box(stacked_image, (x1,y1,x2,y2), box_to_merge[1], str(key), font_scale=1)
                        x1,y1,x2,y2 = box_to_merge[2]
                        self.draw_bounding_box(stacked_image, (x1,y1,x2,y2), box_to_merge[3], str(key), font_scale=1)
                    else:
                        self.draw_bounding_box(stacked_image, (x1,y1,x2,y2), label, str(key), font_scale=1)
            except Exception as e:
                print("Error in merging boxes for key:", key, "with box_to_merge:", box_to_merge, "err:", e)

        # 6) Save/show
        if self._is_save_video and self._cap is not None:
            try:
                self._cap.write(stacked_image)
            except Exception as e:
                print("cap.write error:", e)

        if self._is_save_tracking_video and self._tracking_cap is not None:
            try:
                self._tracking_cap.write(stacked_image)
            except Exception as e:
                print("tracking_cap.write error:", e)

        if self._show:
            try:
                imshow_size = (576, int((stacked_image.shape[0] / 2)))
                cv2.imshow("Cattle Pipeline (queued)", cv2.resize(stacked_image, imshow_size))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._user_quit = True
                    self._safe_shutdown()
            except Exception as e:
                print("imshow error:", e)

        return stacked_image

    def _safe_shutdown(self):
        """Signal all workers to stop (no GUI teardown here).
        Idempotent and non-blocking.
        """
        if getattr(self, "_is_shutting_down", False):
            return
        self._is_shutting_down = True
        try:
            print("Initiating safe shutdown...")
            if hasattr(self, "_stop_ev") and self._stop_ev is not None:
                self._stop_ev.set()
            # deliver STOPs to unblock queues/consumers without blocking
            for q in (getattr(self, "_q_read", None), getattr(self, "_q_detect", None)):
                try:
                    q_put_stop(q)
                except Exception as ex:
                    print("q_put_stop error:", ex)
        finally:
            # Do NOT call cv2.destroyAllWindows() here; the track thread owns UI
            print("one video finish")
