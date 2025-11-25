"""
Database Schema and Manager
Project ID: Image Processing App 20251119
Created: 2025-01-19 06:46:25 UTC
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path
import logging

Base = declarative_base()


class ProcessingJob(Base):
    """Main processing job tracking table."""
    __tablename__ = 'processing_jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), unique=True, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed, cancelled
    menu_option = Column(Integer, nullable=False)
    source_paths = Column(JSON, nullable=False)
    output_path = Column(String(500), nullable=False)
    admin_path = Column(String(500), nullable=False)
    total_images = Column(Integer, default=0)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    error_message = Column(Text)
    checkpoint_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ImageRecord(Base):
    """Individual image processing records."""
    __tablename__ = 'image_records'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(36), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer)
    file_hash = Column(String(64))
    
    # Processing status
    blur_detected = Column(Boolean)
    blur_score = Column(Float)
    metadata_extracted = Column(Boolean)
    caption_generated = Column(Boolean)
    colors_analyzed = Column(Boolean)
    
    # Transformation status
    grayscale_created = Column(Boolean)
    sepia_created = Column(Boolean)
    pencil_sketch_created = Column(Boolean)
    coloring_book_created = Column(Boolean)
    connect_dots_created = Column(Boolean)
    color_by_numbers_created = Column(Boolean)
    
    # Timing
    processing_start = Column(DateTime)
    processing_end = Column(DateTime)
    processing_duration = Column(Float)  # seconds
    
    # Error tracking
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ImageMetadata(Base):
    """Extended metadata storage."""
    __tablename__ = 'image_metadata'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    
    # EXIF data
    camera_make = Column(String(100))
    camera_model = Column(String(100))
    datetime_original = Column(DateTime)
    datetime_digitized = Column(DateTime)
    datetime_modified = Column(DateTime)
    
    # Image properties
    width = Column(Integer)
    height = Column(Integer)
    orientation = Column(String(20))  # portrait, landscape, square
    aspect_ratio = Column(String(20))  # 16:9, 4:3, etc.
    
    # GPS data
    gps_latitude = Column(Float)
    gps_longitude = Column(Float)
    gps_altitude = Column(Float)
    gps_location_readable = Column(String(500))
    
    # Calculated fields
    capture_season_met = Column(String(20))  # meteorological season
    capture_season_astro = Column(String(20))  # astronomical season
    capture_quarter = Column(String(10))  # Q1, Q2, Q3, Q4
    capture_time_of_day = Column(String(20))  # morning, afternoon, evening, night
    capture_decade = Column(String(10))  # 2020s, etc.
    
    # AI-generated content
    primary_caption = Column(Text)
    primary_description = Column(Text)
    keywords = Column(JSON)
    tags = Column(JSON)
    alt_text = Column(Text)
    
    # Color analysis
    dominant_colors = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self, db_path: Path, config: dict, logger: logging.Logger):
        self.db_path = db_path
        self.config = config
        self.logger = logger
        self.engine = None
        self.Session = None
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection and create tables."""
        db_file = self.db_path / self.config.get('database', {}).get('db_name', 'image_processing.db')
        db_url = f"sqlite:///{db_file}"
        
        self.logger.info(f"Initializing database: {db_file}")
        
        self.engine = create_engine(
            db_url,
            echo=self.config.get('database', {}).get('echo_sql', False),
            pool_size=self.config.get('database', {}).get('connection_pool_size', 5)
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        self.logger.info("Database initialized successfully")
    
    def create_job(self, job_data: dict) -> str:
        """Create a new processing job."""
        import uuid
        
        session = self.Session()
        try:
            job = ProcessingJob(
                job_id=str(uuid.uuid4()),
                **job_data
            )
            session.add(job)
            session.commit()
            self.logger.info(f"Created processing job: {job.job_id}")
            return job.job_id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to create job: {e}")
            raise
        finally:
            session.close()
    
    def update_job_progress(self, job_id: str, processed: int, failed: int = 0):
        """Update job progress."""
        session = self.Session()
        try:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()
            if job:
                job.processed_images = processed
                job.failed_images = failed
                job.updated_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to update job progress: {e}")
        finally:
            session.close()
    
    def save_checkpoint(self, job_id: str, checkpoint_data: dict):
        """Save checkpoint data for resume capability."""
        session = self.Session()
        try:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()
            if job:
                job.checkpoint_data = checkpoint_data
                job.updated_at = datetime.utcnow()
                session.commit()
                self.logger.debug(f"Checkpoint saved for job {job_id}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save checkpoint: {e}")
        finally:
            session.close()
    
    def load_checkpoint(self, job_id: str) -> dict:
        """Load checkpoint data for resume."""
        session = self.Session()
        try:
            job = session.query(ProcessingJob).filter_by(job_id=job_id).first()
            return job.checkpoint_data if job else {}
        finally:
            session.close()