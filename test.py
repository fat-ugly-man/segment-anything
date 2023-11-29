import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam_checkpoint2 = "models/sam_vit_l_0b3195.pth"
model_type2 = "vit_l"

sam_checkpoint3 = "models/sam_vit_h_4b8939.pth"
model_type3 = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

sam2 = sam_model_registry[model_type2](checkpoint=sam_checkpoint2)
sam2.to(device=device)

sam3 = sam_model_registry[model_type3](checkpoint=sam_checkpoint3)
sam3.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16, #控制采样点的间隔，值越小，采样点越密集
    pred_iou_thresh=0.8, #mask的iou阈值
    stability_score_thresh=0.9, #mask的稳定性阈值
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  #最小mask面积，会使用opencv滤除掉小面积的区域
)

mask_generator2 = SamAutomaticMaskGenerator(
    model=sam2,
    points_per_side=16, #控制采样点的间隔，值越小，采样点越密集
    pred_iou_thresh=0.8, #mask的iou阈值
    stability_score_thresh=0.9, #mask的稳定性阈值
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  #最小mask面积，会使用opencv滤除掉小面积的区域
)

mask_generator3 = SamAutomaticMaskGenerator(
    model=sam3,
    points_per_side=16, #控制采样点的间隔，值越小，采样点越密集
    pred_iou_thresh=0.8, #mask的iou阈值
    stability_score_thresh=0.9, #mask的稳定性阈值
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  #最小mask面积，会使用opencv滤除掉小面积的区域
)

def show_nums(w, h, dis1, dis2):
    CI = w / h
    CVA = abs(dis1 - dis2)
    CVAI = CVA / max(dis1, dis2) * 100
    CI_tag = ''
    CVAI_tag = ''
    if CI <= 0.6:
        CI_tag = '舟状头（重度）'
    elif CI > 0.6 and CI <= 0.7:
        CI_tag = '舟状头（中度）'
    elif CI > 0.7 and CI <= 0.75:
        CI_tag = '舟状头（轻度）'
    elif CI > 0.75 and CI < 0.85:
        CI_tag = '正常（无扁头，无舟状头）'
    elif CI >= 0.85 and CI < 0.9:
        CI_tag = '扁头（轻度）'
    elif CI >= 0.9 and CI < 0.99:
        CI_tag = '扁头（中度）'
    else:
        CI_tag = '扁头（重度）'

    if CVAI < 3.5:
        CVAI_tag = '正常（无斜头）'
    elif CVAI >= 3.5 and CVAI < 6.25:
        CVAI_tag = '斜头畸形（轻度）'
    elif CVAI >= 6.25 and CVAI < 8.75:
        CVAI_tag = '斜头畸形（中度）'
    elif CVAI >= 8.75 and CVAI < 11:
        CVAI_tag = '斜头畸形（重度）'
    elif CVAI >= 11:
        CVAI_tag = '斜头畸形（超重度）'

    print ('CI:{}'.format(CI))
    print ('CI测试结果:{}'.format(CI_tag))
    print ('CVAI:{}'.format(CVAI))
    print ('CVAI测试结果:{}'.format(CVAI_tag))

# 定义旋转函数
def rotate_line(img, cnt, cx, cy, angle, length):
    # 计算旋转矩阵
    #M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    # 旋转图像
    #img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    original_h = img.shape[0]
    original_w = img.shape[1]
    while True:
        x1 = cx - math.sin(angle) * length / 2
        y1 = cy + math.cos(angle) * length / 2
        x2 = cx + math.sin(angle) * length / 2
        y2 = cy - math.cos(angle) * length / 2
        if x1 < 0 or x1 > original_w or x2 < 0 or x2 > original_w:
            length -= 1
            continue
        if y1 < 0 or y1 > original_h or y2 < 0 or y2 > original_h:
            length -= 1
            continue
        break
    
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    # 绘制旋转后的线段
    #cv2.line(img, pt1, pt2, (255, 255, 255), 2)
   
    final = []
    flag = False
    intersections = []
    cha = []
    line_cnt = np.array([pt1, pt2])
   
    for pt in cnt:
        pt = (int(pt[0][0]), int(pt[0][1]))
        dist = cv2.pointPolygonTest(line_cnt, pt, True)
        if abs(dist) < 20:
            flag = True
            intersections.append(pt)
            cha.append(abs(dist))
        else:
            if len(intersections) > 0:
                min_cha = min(cha)
                min_index = cha.index(min_cha)
                final.append(intersections[min_index])
                #cv2.circle(img, intersections[min_index], 5, (255, 255, 0), -1) 
            flag = False
            intersections = []
            cha = []
    if len(final) == 2:
        cv2.line(img, final[0], final[1], (255, 255, 255), 2)
        dis = math.sqrt((final[0][0] - final[1][0])*(final[0][0] - final[1][0]) + (final[0][1] - final[1][1])*(final[0][1] - final[1][1]))
        return img, dis
    else:
        return img, -1

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    contour_all = []

    pos = 0
    while True:
        hh = sorted_anns[pos]['bbox'][3] / image.shape[0]
        ww = sorted_anns[pos]['bbox'][2] / image.shape[1]
        if hh < 0.95 and ww < 0.95:
            break
        pos += 1
    #print (image.shape)
    #print (sorted_anns[pos]['bbox'])
    max_err = 3
    now_bbox = sorted_anns[pos]['bbox']
    if now_bbox[0] < max_err or now_bbox[1] < max_err:
        return False
    if abs(now_bbox[0] + now_bbox[2] - image.shape[1]) < max_err or abs(now_bbox[1] + now_bbox[3] - image.shape[0]) < max_err:
        return False

    m = sorted_anns[pos]['segmentation']
    contours, hierarchy = cv2.findContours(m.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    contour_all.append(cnts[0])
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask
    
    original_h = image.shape[0]
    original_w = image.shape[1]
    temp = np.zeros((original_h, original_w, 1))
    cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
    color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
    contour_mask = temp / 255 * color.reshape(1, 1, -1)
    plt.imshow(contour_mask)
    
    temp = np.zeros((original_h, original_w, 1))
    #cv2.rectangle(temp, (now_bbox[0], now_bbox[1]), (now_bbox[0] + now_bbox[2], now_bbox[1] + now_bbox[3]), (255, 255, 255), 2)
    
    scnt1 = sorted(cnts[0], key= lambda x:x[0][0])
    cv2.line(temp, (scnt1[0][0][0], scnt1[0][0][1]), (scnt1[-1][0][0], scnt1[-1][0][1]), (255, 255, 255), 2)
    
    scnt2 = sorted(cnts[0], key= lambda x:x[0][1])
    cv2.line(temp, (scnt2[0][0][0], scnt2[0][0][1]), (scnt2[-1][0][0], scnt2[-1][0][1]), (255, 255, 255), 2)

    # 计算垂直方向的线段的中点和角度
    x1, y1, x2, y2 = scnt2[0][0][0], scnt2[0][0][1], scnt2[-1][0][0], scnt2[-1][0][1] # 垂直方向的线段的起点和终点
    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2) # 中点
    angle = math.atan2(y2 - y1, x2 - x1) # 角度
    length = int(math.sqrt(now_bbox[2] * now_bbox[2] + now_bbox[3] * now_bbox[3]))

    # 旋转线段30度
    temp, dis1 = rotate_line(temp, cnts[0], cx, cy, angle + math.radians(60), length * 2)
    # 旋转线段-30度
    temp, dis2 = rotate_line(temp, cnts[0], cx, cy, angle - math.radians(60), length * 2)
    
    if dis1 == -1 or dis2 == -1:
        return False
    color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
    rectangle_mask = temp / 255 * color.reshape(1, 1, -1)
    plt.imshow(rectangle_mask)
    
    show_nums(now_bbox[2], now_bbox[3], dis1, dis2)
    #ax.imshow(img)
    return True

models = [sam2]
ious = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
stability_scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
for root, dirs, files in os.walk("./images/"):
    for file_name in files:
        #if file_name != '23.png':
        #    continue
        print (file_name)
        image = cv2.imread('images/' + file_name)
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(16,16))
        plt.imshow(image)
        
        masks3 = mask_generator3.generate(image)
        suc_state = show_anns(masks3)
        if not suc_state:
            masks2 = mask_generator2.generate(image)
            suc_state = show_anns(masks2)
            if not suc_state:
                masks = mask_generator.generate(image)
                suc_state = show_anns(masks)
        
        if not suc_state:
            print ('识别失败！')
            continue
        plt.axis('off')
        #plt.show()
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
        result = result[:, :, ::-1]
        cv2.imwrite('output/' + file_name, result)
