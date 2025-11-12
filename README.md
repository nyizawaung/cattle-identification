# CATTLE IDENTIFICATION
Detection with detctron 2
Customized tracking with centroid based on center of mask location
Taking max predicted ID from batch identitication for reducing system run time, increaseing the accuracy and stablizing the ID switching problem

# MULTIPLE CAMERA TRACKING
To identify and tracked cattle on multiple cameras that are vertically stacked and not aligned
Cameras are fixed manually to make it aligned. Purpose is to carry the tracking id from one camera to another camera and reduce the identification time

# KNP NEW TRACKING queue Quartic Regression
This is prototype version of multiple camera tracking without cutting and concatting the camera views.
The purpose is to calculate the can-be location in next camera using the Quartic regression from the Y(centroid)

# QUEUE IN FILE NAME
Running multiple camera will take toll in the system performace. for that reason, using parallel processing with 3 queues, reading, detection and tracking-identification to run in real time. Can run up to 3 videos.
System can be faster if there is no need to save crop image (to collect identification dataset), 
Reduce extra info in getBatchIdentification, e.g to count the missed frame 

# KNP_Single_Cam_ReCheck
To get the final cattle Id, first do the 10 batches of identification. semi-final
then do one batch after cattle move to 100 pixels. if system get the same ID, it is considered as final cattle Id otherwise reset the data and do identification from beginning. -> ReCheck mean, there is no final cattle id,
system will do the cattle identification as long as the cattle move 100 pixels.

# Dataset collection
 Data collection is important in this identification system because cattle can be in multiple postures and each postures have different pattern for each cow. To get a better accuracy, it is neccessary to collect multiple postures of the object.
