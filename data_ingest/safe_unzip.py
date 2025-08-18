"""Safe unzipping utilities for UMAFall dataset."""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SafeUnzipper:
    """Safely unzip dataset files with validation."""
    
    def __init__(self, max_size_gb: float = 10.0):
        """
        Initialize safe unzipper.
        
        Args:
            max_size_gb: Maximum allowed uncompressed size in GB
        """
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    
    def unzip(self, zip_path: Path, extract_to: Path,
              password: Optional[str] = None) -> bool:
        """
        Safely unzip a file with size validation.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            password: Optional password for encrypted zip
            
        Returns:
            True if successful
        """
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            return False
        
        # Validate zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check for zip bombs
                total_size = sum(info.file_size for info in zf.infolist())
                compressed_size = sum(info.compress_size for info in zf.infolist())
                
                if total_size > self.max_size_bytes:
                    logger.error(
                        f"Uncompressed size ({total_size / 1e9:.2f} GB) exceeds "
                        f"maximum ({self.max_size_bytes / 1e9:.2f} GB)"
                    )
                    return False
                
                # Check compression ratio (potential zip bomb)
                if compressed_size > 0:
                    ratio = total_size / compressed_size
                    if ratio > 100:
                        logger.warning(
                            f"High compression ratio ({ratio:.1f}x) detected. "
                            f"Potential zip bomb?"
                        )
                
                # Check for path traversal
                for info in zf.infolist():
                    if self._is_path_traversal(info.filename):
                        logger.error(
                            f"Path traversal detected in: {info.filename}"
                        )
                        return False
                
                # Extract files
                logger.info(f"Extracting {len(zf.infolist())} files...")
                extract_to.mkdir(parents=True, exist_ok=True)
                
                if password:
                    zf.setpassword(password.encode())
                
                zf.extractall(extract_to)
                logger.info(f"Successfully extracted to {extract_to}")
                return True
                
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error extracting zip: {e}")
            return False
    
    def _is_path_traversal(self, filename: str) -> bool:
        """
        Check if filename contains path traversal.
        
        Args:
            filename: Filename from zip archive
            
        Returns:
            True if path traversal detected
        """
        # Check for absolute paths
        if os.path.isabs(filename):
            return True
        
        # Check for .. in path
        parts = Path(filename).parts
        if '..' in parts:
            return True
        
        # Check for leading slashes
        if filename.startswith('/') or filename.startswith('\\'):
            return True
        
        return False