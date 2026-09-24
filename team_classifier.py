import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_jersey_pixels(frame, bbox):
    """
    Crops the player torso and removes green pitch grass using HSV thresholding.
    Returns array of BGR jersey pixels.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Grab top 50% of bounding box to isolate the shirt
    height = y2 - y1
    torso_y2 = y1 + int(height * 0.5)
    torso = frame[y1:torso_y2, x1:x2]
    
    # Skip invalid/empty crops (e.g. bounding box at frame edge)
    if torso.size == 0:
        return None

    # Convert crop to HSV to handle lighting variations
    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    
    # Grass green color bounds in OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Mask pitch grass out and invert to keep jersey
    green_mask = cv2.inRange(hsv_torso, lower_green, upper_green)
    jersey_mask = cv2.bitwise_not(green_mask)
    
    # Filter matrix down to non-green pixel values
    jersey_pixels = torso[jersey_mask > 0]
    return jersey_pixels


def get_dominant_color(jersey_pixels):
    """
    Extracts the dominant BGR color vector from shirt pixels using KMeans (k=1).
    """
    if len(jersey_pixels) == 0:
        return np.array([0, 0, 0])
        
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(jersey_pixels)
    return kmeans.cluster_centers_[0].astype(int)


def fit_team_classifier(all_player_colors):
    """
    Fits global 2-cluster KMeans model on collected player colors to define the two teams.
    """
    if len(all_player_colors) < 2:
        return None
        
    team_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    team_kmeans.fit(all_player_colors)
    return team_kmeans


def assign_team(player_color, team_kmeans):
    """
    Predicts team assignment (0 or 1) and returns corresponding team BGR cluster center.
    """
    team_id = team_kmeans.predict([player_color])[0]
    team_color = team_kmeans.cluster_centers_[team_id].astype(int)
    return team_id, team_color
