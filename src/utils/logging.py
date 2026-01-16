"""
Logging Module for TEKNOFEST Savaşan İHA Simulation.

Provides centralized, configurable logging with:
- Console and file handlers
- Structured log format
- Component-specific loggers
- Performance timing decorators
"""

import logging
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional
import time


# Color codes for terminal output
class LogColors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal."""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        
    def format(self, record):
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.CYAN}{record.name}{LogColors.RESET}"
        return super().format(record)


class SimulationLogger:
    """Centralized logging manager for simulation."""
    
    _instance = None
    _initialized = False
    
    # Logger hierarchy
    ROOT_LOGGER = "simulation"
    COMPONENT_LOGGERS = [
        "simulation.core",
        "simulation.vision",
        "simulation.uav",
        "simulation.competition",
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SimulationLogger._initialized:
            return
        SimulationLogger._initialized = True
        
        self.root_logger = logging.getLogger(self.ROOT_LOGGER)
        self.root_logger.setLevel(logging.DEBUG)
        self._file_handler = None
        self._console_handler = None
        
    def setup(
        self,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        console: bool = True,
        use_colors: bool = True,
    ):
        """
        Configure logging.
        
        Args:
            level: Minimum log level (logging.DEBUG, INFO, etc.)
            log_file: Optional file path for persistent logs
            console: Enable console output
            use_colors: Use colored console output
        """
        # Clear existing handlers
        self.root_logger.handlers.clear()
        
        # Log format
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%H:%M:%S"
        
        # Console handler
        if console:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(level)
            self._console_handler.setFormatter(
                ColoredFormatter(log_format, date_format, use_colors)
            )
            self.root_logger.addHandler(self._console_handler)
            
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            self._file_handler = logging.FileHandler(log_file, encoding='utf-8')
            self._file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            self._file_handler.setFormatter(
                logging.Formatter(log_format, date_format)
            )
            self.root_logger.addHandler(self._file_handler)
            
        self.root_logger.setLevel(level)
        
    def get_logger(self, name: str) -> logging.Logger:
        """Get a named logger under the simulation hierarchy."""
        if not name.startswith(self.ROOT_LOGGER):
            name = f"{self.ROOT_LOGGER}.{name}"
        return logging.getLogger(name)
    
    def set_level(self, level: int):
        """Change log level dynamically."""
        self.root_logger.setLevel(level)
        if self._console_handler:
            self._console_handler.setLevel(level)


# Module-level convenience functions
_logger_manager = SimulationLogger()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    use_colors: bool = True,
):
    """Setup logging for the simulation."""
    _logger_manager.setup(level, log_file, console, use_colors)


def get_logger(name: str) -> logging.Logger:
    """Get a component logger."""
    return _logger_manager.get_logger(name)


# Pre-configured component loggers
def get_core_logger() -> logging.Logger:
    return get_logger("core")


def get_vision_logger() -> logging.Logger:
    return get_logger("vision")


def get_uav_logger() -> logging.Logger:
    return get_logger("uav")


def get_competition_logger() -> logging.Logger:
    return get_logger("competition")


# Performance timing decorator
def timed(logger: logging.Logger = None, level: int = logging.DEBUG):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            log = logger or get_logger("timing")
            log.log(level, f"{func.__name__} completed in {elapsed:.2f}ms")
            
            return result
        return wrapper
    return decorator


# Context manager for timed blocks
class TimedBlock:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: logging.Logger = None, level: int = logging.DEBUG):
        self.name = name
        self.logger = logger or get_logger("timing")
        self.level = level
        self.start = None
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000
        self.logger.log(self.level, f"{self.name} completed in {elapsed:.2f}ms")
