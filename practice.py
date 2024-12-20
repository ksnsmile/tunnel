# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:58:40 2024

@author: USER
"""

# 필요한 라이브러리 임포트
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 이미지 파일 경로
image_path = 'image/0-0.png'

# 이미지 읽기 (그레이스케일)
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지를 1차원으로 변환
# -1은 이 배열의 크기를 계산, 1은 두번째 차원의 크기
pixels = original_image.reshape(-1, 1)

# K-평균 클러스터링
k = 3  # 클러스터의 수
kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
# 클러스터 레이블을 원본 이미지 크기로 변환
segmented_image = kmeans.labels_.reshape(original_image.shape)
# 오츠 알고리즘을 사용한 이진화
otsu_threshold, binary_image = cv2.threshold(segmented_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# num_labesl 덩어리 개수(배경 포함), ㅣlabels는 덩어리가 3개면 0,1,2 
# 작은 구성 요소 제거 함수 정의
def remove_small_components(binary_image):
    num_labels, labels = cv2.connectedComponents(binary_image)
    output_image = np.zeros_like(binary_image)
    component_sizes = []

    # 각 연결 성분의 크기 계산
    for label in range(1, num_labels):
        component_mask = (labels == label)
        component_size = np.sum(component_mask)
        component_sizes.append(component_size)

    # 이상치 제거를 위한 Z-스코어 계산
    if component_sizes:
        mean_size = np.mean(component_sizes)
        std_size = np.std(component_sizes)

        # Z-스코어가 특정 임계값을 넘는 성분만 유지 (예: |Z| < 2)
        threshold = 2  # 필요에 따라 조정 가능
        filtered_sizes = [size for size in component_sizes if abs((size - mean_size) / std_size) < threshold]

        # 이상치를 제거한 후 평균 크기 계산
        if filtered_sizes:
            filtered_mean_size = np.mean(filtered_sizes)
        else:
            filtered_mean_size = 0

        # 이상치를 제거한 후 평균 크기보다 큰 성분만 남기기
        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)
            if component_size > filtered_mean_size:
                output_image[component_mask] = 255
    else:
        filtered_mean_size = 0

    return output_image, filtered_mean_size

output_image, average_size = remove_small_components(binary_image)

# 결과를 화면에 표시
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

# K-평균 클러스터링 결과
plt.subplot(1, 4, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

# 오츠 이진화된 이미지
plt.subplot(1, 4, 3)
plt.title('Binary Image (Otsu)')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

# 작은 구성 요소 제거 결과
plt.subplot(1, 4, 4)
plt.title('Post-processed Image')
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.show()


# 윤곽선 검출
contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선 그리기
contour_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# 각 컨투어의 평균 화소값 계산 및 정보 저장
contour_data = []
mean_pixel_values = []

for i, contour in enumerate(contours):
    mask = np.zeros_like(output_image)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # 컨투어를 흰색으로 채운 마스크 생성
    mean_val = cv2.mean(original_image, mask=mask)[0]  # 마스크 영역의 평균 화소값 계산 (원본 이미지 사용)
    mean_pixel_values.append(mean_val)
    
    # 컨투어의 중심점 계산
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x, center_y = contour[0][0]
    
    # 컨투어 데이터 저장
    contour_info = {
        "contour_index": i,
        "contour_points": contour.tolist(),
        "mean_pixel_value": mean_val,
        "center": [center_x, center_y]
    }
    contour_data.append(contour_info)

# 결과 시각화
plt.figure(figsize=(12, 6))

# 원본 이진화 이미지
plt.subplot(1, 2, 1)
plt.imshow(output_image, cmap='gray')
plt.title('Binary Image')

# 윤곽선 이미지
plt.subplot(1, 2, 2)
plt.imshow(contour_image)
plt.title('Contours')

plt.show()

# 평균 화소값 출력
for i, mean_val in enumerate(mean_pixel_values):
    print(f"Contour {i}: Mean pixel value = {mean_val}")

# contour_data 리스트에 컨투어 정보가 저장됨

import json

def calculate_pca(contour):
    data_pts = np.array(contour, dtype=np.float64).reshape(-1, 2)
    mean = np.mean(data_pts, axis=0)
    centered_data_pts = data_pts - mean
    cov_matrix = np.cov(centered_data_pts, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    major_axis = eigenvectors[:, -1]
    angle = np.arctan2(major_axis[1], major_axis[0])
    return angle

def extend_line_within_bounds(center, angle, x_min, x_max, y_min, y_max):
    direction = np.array([np.cos(angle), np.sin(angle)])
    scale_factors = []
    if direction[0] != 0:
        scale_factors.extend([
            (x_min - center[0]) / direction[0],
            (x_max - center[0]) / direction[0]
        ])
    if direction[1] != 0:
        scale_factors.extend([
            (y_min - center[1]) / direction[1],
            (y_max - center[1]) / direction[1]
        ])
    if not scale_factors:
        return tuple(center), tuple(center)
    positive_scales = [sf for sf in scale_factors if sf > 0]
    negative_scales = [sf for sf in scale_factors if sf < 0]
    min_scale = max(negative_scales) if negative_scales else 0
    max_scale = min(positive_scales) if positive_scales else 1
    start_point = center + direction * min_scale
    end_point = center + direction * max_scale
    start_point = np.clip(start_point, [x_min, y_min], [x_max, y_max])
    end_point = np.clip(end_point, [x_min, y_min], [x_max, y_max])
    return tuple(start_point.astype(int)), tuple(end_point.astype(int))

def draw_gradients_on_image(image, contours):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pca_angles = []
    centers = []
    lines_info = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = tuple(contour[0][0])
        centers.append((center_x, center_y))
        angle = calculate_pca(contour)
        pca_angles.append(angle)
        contour_array = np.array(contour).reshape(-1, 2)
        x_min, y_min = contour_array.min(axis=0)
        x_max, y_max = contour_array.max(axis=0)
        start_point, end_point = extend_line_within_bounds(
            np.array([center_x, center_y]), angle, 
            x_min, x_max, y_min, y_max
        )
        cv2.line(color_image, start_point, end_point, (0, 255, 0), 2)
        lines_info.append({
            'start': start_point,
            'end': end_point,
            'center': (center_x, center_y),
        })
    return color_image, pca_angles, centers, lines_info

# 주성분선 그리기 및 주성분선 기울기 및 중심점 계산
gradient_image, pca_angles, centers, lines_info = draw_gradients_on_image(output_image, contours)

# 결과 시각화
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Image with Major Axes')
plt.imshow(gradient_image)
plt.axis('off')
plt.show()

# 평균 화소값 및 주성분선 기울기 및 중심점 좌표 출력
for i, (mean_val, angle, center) in enumerate(zip(mean_pixel_values, pca_angles, centers)):
    print(f"Contour {i}: Mean pixel value = {mean_val}, PCA angle (radians) = {angle}, Center = {center}")

# 컨투어 데이터를 JSON 형식으로 변환
cluster_data = {
    'contours': [
        {
            'index': i,
            'mean_pixel_value': mean_val,
            'pca_angle': angle,
            'contour': contour.tolist(),
            'line': {
                'start': lines_info[i]['start'],
                'end': lines_info[i]['end'],
                'center': lines_info[i]['center'],
            }
        }
        for i, (mean_val, angle, center) in enumerate(zip(mean_pixel_values, pca_angles, centers))
    ]
}

# JSON 파일로 저장
def convert_np_types(o):
    if isinstance(o, (np.int32, np.int64)): return int(o)
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError

json_file_path = 'contours_data.json'
with open(json_file_path, 'w') as json_file:
    json.dump(cluster_data, json_file, indent=4, default=convert_np_types)

print(f"Data has been saved to {json_file_path}")



def draw_gradients_on_image(image, contours):
    """
    Draw the major axis direction on the image for each contour.
    
    :param image: Input grayscale image.
    :param contours: List of contours.
    :return: Image with gradients drawn and list of PCA angles, centers, start and end points.
    """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pca_angles = []  # List to store PCA angles for each contour
    centers = []  # List to store center coordinates for each contour
    lines_info = []  # List to store start and end points for each line

    for contour in contours:
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = tuple(contour[0][0])
        
        centers.append((center_x, center_y))

        # Calculate the major axis angle using PCA
        angle = calculate_pca(contour)
        pca_angles.append(angle)  # Save the PCA angle

        length = 100  # Length of the line to draw
        
        start_x = int(center_x - length * np.cos(angle))
        start_y = int(center_y - length * np.sin(angle))
        
        end_x = int(center_x + length * np.cos(angle))
        end_y = int(center_y + length * np.sin(angle))

        cv2.line(color_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Store the start, end, and center points
        lines_info.append({
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'center': (center_x, center_y),
            'angle': angle
        })

    return color_image, lines_info

def group_lines_by_angle(lines, threshold_angle=np.radians(45)):
    x_axis_lines = []
    y_axis_lines = []
    
    for line in lines:
        angle = line['angle']
        if abs(angle) < threshold_angle or abs(angle) > (np.pi - threshold_angle):
            x_axis_lines.append(line)
        else:
            y_axis_lines.append(line)
    
    return x_axis_lines, y_axis_lines

def draw_lines_on_image(image, lines, color):
    """
    Draw lines on the image.
    :param image: Input image.
    :param lines: List of line information (start, end, center, angle).
    :param color: Color of the lines (BGR format).
    :return: Image with lines drawn.
    """
    if len(image.shape) == 2:  # If the image is grayscale
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color_image = image.copy()
    
    for line in lines:
        start, end, center, _ = line.values()
        cv2.line(color_image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 2)
        cv2.circle(color_image, (int(center[0]), int(center[1])), 5, color, -1)  # Mark the center

    return color_image

# Assuming output_image and contours are already defined
gradient_image, lines_info = draw_gradients_on_image(output_image, contours)

# Group lines by angle
x_axis_lines, y_axis_lines = group_lines_by_angle(lines_info)

# Draw grouped lines on the image
x_axis_image = draw_lines_on_image(output_image, x_axis_lines, (0, 255, 0))  # Red
y_axis_image = draw_lines_on_image(output_image, y_axis_lines, (0, 255, 0))  # Green

# Display results
plt.figure(figsize=(15, 10))

# X-axis lines
plt.subplot(1, 2, 1)
plt.title('X-axis Lines')
plt.imshow(cv2.cvtColor(x_axis_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.axis('off')

# Y-axis lines
plt.subplot(1, 2, 2)
plt.title('Y-axis Lines')
plt.imshow(cv2.cvtColor(y_axis_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.axis('off')

plt.show()















