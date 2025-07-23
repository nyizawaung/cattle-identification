from pathlib import Path

class Helper:
    
    def increment_path(path, exist_ok=False, sep='', mkdir=False):
        """
        Increment path, i.e. runs/exp, runs/exp1, runs/exp2, etc.
        
        Args:
            path (str or Path): The base path.
            exist_ok (bool): Whether to return the original path if it exists.
            sep (str): Separator between the base path and the incrementing number.
            mkdir (bool): Whether to create the directory for the new path.
        
        Returns:
            Path: Incremented path.
        """
        path = Path(path)
        if path.exists() and exist_ok:
            return path
        else:
            # Extract the base name and suffix
            dir = path.parent
            base = path.stem
            suffix = path.suffix
            
            # Initialize the new path
            new_path = path
            
            # Increment the path if necessary
            for i in range(1, 9999):
                new_path = Path(f"{dir}/{base}{sep}{i}{suffix}")
                if not new_path.exists():
                    break
            
            if mkdir:
                new_path.mkdir(parents=True, exist_ok=True)
            
            return new_path

# Example usage:

