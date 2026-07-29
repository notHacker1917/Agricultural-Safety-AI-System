import sqlite3
import logging
import os
import tempfile
import numpy as np

class TrajectoryStorage:
    """
    Store trajectories in SQLite.
    """
    def __init__(self, db_path=None):
        """
        Initialize database.

        Args:
            db_path (str): Path to SQLite DB. If None, uses temp dir.
        """
        if db_path is None:
            db_path = os.path.join(tempfile.gettempdir(), 'trajectories.db')
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        logging.info(f"Trajectory storage initialized at {db_path}")

    def create_table(self):
        """
        Create trajectories table.
        """
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trajectories (
                frame_id INTEGER,
                object_id INTEGER,
                centroid_x REAL,
                centroid_y REAL,
                mask_area REAL
            )
        ''')
        self.conn.commit()

    def store(self, frame_id, object_id, centroid, mask_area):
        """
        Store trajectory data.

        Args:
            frame_id (int): Frame ID.
            object_id (int): Object ID.
            centroid (tuple): (x, y)
            mask_area (float): Area of mask.
        """
        self.conn.execute('''
            INSERT INTO trajectories (frame_id, object_id, centroid_x, centroid_y, mask_area)
            VALUES (?, ?, ?, ?, ?)
        ''', (frame_id, object_id, centroid[0], centroid[1], mask_area))
        self.conn.commit()
        logging.debug(f"Stored trajectory for object {object_id} at frame {frame_id}")

    def get_trajectory(self, object_id, last_n=30, smooth=True):
        """
        Get last n points for object, optionally smoothed.

        Args:
            object_id (int): Object ID.
            last_n (int): Number of points.
            smooth (bool): Apply moving average smoothing.

        Returns:
            list: List of (x, y)
        """
        cursor = self.conn.execute('''
            SELECT centroid_x, centroid_y FROM trajectories
            WHERE object_id = ?
            ORDER BY frame_id DESC
            LIMIT ?
        ''', (object_id, last_n))
        points = [(row[0], row[1]) for row in cursor.fetchall()]
        
        if smooth and len(points) > 2:
            smoothed = []
            for i in range(len(points)):
                if i == 0:
                    smoothed.append(points[0])
                elif i == len(points) - 1:
                    smoothed.append(points[-1])
                else:
                    x = (points[i-1][0] + points[i][0] + points[i+1][0]) / 3
                    y = (points[i-1][1] + points[i][1] + points[i+1][1]) / 3
                    smoothed.append((x, y))
            points = smoothed
        
        return points

    def get_velocity(self, object_id):
        """
        Get velocity vector (dx, dy) from last two points.

        Args:
            object_id (int): Object ID.

        Returns:
            tuple: (dx, dy) or (0, 0) if not enough points.
        """
        cursor = self.conn.execute('''
            SELECT centroid_x, centroid_y FROM trajectories
            WHERE object_id = ?
            ORDER BY frame_id DESC
            LIMIT 2
        ''', (object_id,))
        rows = cursor.fetchall()
        if len(rows) < 2:
            return (0, 0)
        curr = np.array(rows[0])
        prev = np.array(rows[1])
        velocity = curr - prev
        return (velocity[0], velocity[1])

    def update(self, frame_id, object_id, bbox):
        """
        Update trajectory storage for the current object.

        Args:
            frame_id (int): Current frame index.
            object_id (int): Object ID.
            bbox (tuple): Bounding box (x1, y1, x2, y2).
        """
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        mask_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.store(frame_id, object_id, centroid, mask_area)

    def reset(self):
        """
        Reset trajectory storage by clearing stored trajectories.
        """
        self.conn.execute('DELETE FROM trajectories')
        self.conn.commit()
        logging.info("Trajectory storage reset")

    def close(self):
        """
        Close database connection.
        """
        self.conn.close()
        logging.info("Trajectory storage closed")