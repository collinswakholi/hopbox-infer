# Description: This file contains the functions to compute the features from the images and the results from the YOLOv8-seg model.
from ultralytics.engine.results import Results
import cv2
import numpy as np
import pandas as pd
from readQR import ReadQR
import readQR.wechat_artefacts as artefacts
from ColorCorrectionML import ColorCorrectionML
from typing import Tuple
from easyocr import Reader
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


qr_reader = ReadQR(artefact_path=artefacts.__path__[0])
ocr_reader = Reader(['en'], gpu=True)


"""
------------------------------------------------------------------------------------
Functions to compute the features
------------------------------------------------------------------------------------
""" 
def color_correction(
    result: Results,
    kwargs: dict,
    show: bool = False
) -> Tuple[Results, Tuple[float, float]]: 
    print('\nColor correcting the image...\n')
    orig_img = result.orig_img

    id_cc = get_ids(result, 'ColorCard')[0]
    img,_ = extract_ROI(result, [id_cc], with_bb=True)
    img = img[0]

    if show:
        plt.imshow(img[:,:,::-1])
        plt.title('Color card')
        plt.show()

    cc = ColorCorrectionML(img=img,
                            chart='Classic',
                            illuminant='D50')
    
    _, patch_size = cc.compute_correction(show=show, **kwargs)

    corrected_img = cc.correct_img(orig_img, show=show)
    # corrected_img = cc.Parallel_correct_img(orig_img, show=show, chunks_=900000)
    
    result.orig_img = corrected_img
    print('\nColor correction complete.\n')

    return result, patch_size

def read_QR_code(result: Results,
                 show: bool = False
                 ) -> str:
    print('\nReading QR code...\n')
    QR_info = ''
    ids = get_ids(result, 'QR')
    img,_ = extract_ROI(result, [ids], with_bb=True)
    img = img[0]

    info = qr_reader.decode(img)

    if len(info) < 1:
        print('\033[1;33mUsing OCR to read the QR code...\033[0m')
        # use ocr to read the QR code inforomation using https://github.com/JaidedAI/EasyOCR (pip install easyocr)
        _info_list = ocr_reader.readtext(img, width_ths=0.7, detail=0)

        # Rotate the image and perform OCR to detect text at different angles
        for angle in [90, 270]:
            _info = ocr_reader.readtext(img, detail=0, width_ths=0.7, rotation_info=[angle])
            _info_list.extend(_info)

        # Filter out similar words and keep only unique ones
        unique_words = set()
        for word in _info_list:
            if not any(word in uw for uw in unique_words):
                unique_words.add(word)
        #keep 5 longest words
        unique_words = sorted(unique_words, key=len, reverse=True)[:5]
        _info = list(unique_words)
        print(f'OCR Detected {_info}')

        # Join multiple lines of text with '_' separator
        if len(_info) > 1:
            _info_ = '_'.join([i for i in _info])
            _info = _info_

        QR_info = 'OCR(' + str(_info) +')'
    else:
        QR_info = info

    if show:
        # print the QR code information
        print(f'QR code information: {QR_info}')
        # show the QR code using plt
        plt.imshow(img[:,:,::-1])
        plt.title(f'QR code information: {QR_info}')
        plt.show()
    print('\nQR code reading complete.\n')
    return str(QR_info)

def get_ids(result: Results, class_name: str) -> np.ndarray:
    try:
        return np.array([k for k, v in result.names.items() if v == class_name])
    except:
        print(f'No {class_name} found in the result.')
        return np.array([])

def extract_ROI(result: Results, idx: np.ndarray, with_bb=False) -> Tuple[dict, dict]:
    image = result.orig_img
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    idz = np.where(np.isin(result.boxes.cls.numpy(), idx))[0].tolist()
    imgs = {}
    Poly = {}
    for kn, id in enumerate(idz):
        bb = result.boxes.xyxy[id].data.numpy().astype(int)
        if with_bb:
            imgs[kn] = image[bb[1]:bb[3], bb[0]:bb[2]]
            Poly[kn] = None

        else:    
            poly = result.masks.xy[id].astype(int)
            mask[:] = 0
            mask = cv2.fillPoly(mask, [poly], 255)
            roi = cv2.bitwise_and(image, image, mask=mask)
            cropped = roi[bb[1]:bb[3], bb[0]:bb[2]]
            imgs[kn] = cropped
            Poly[kn] = poly

    return imgs, Poly

def insert_dets(args):
    kn, id, bb, poly, image, names, line_width = args
    img = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), int(line_width*1.5))
    img = cv2.polylines(img, [poly], True, (255, 0, 0),  line_width)

    # insert class name and kn
    text = f'{str(names[id])}_ID_{kn}'
    img = cv2.putText(img, # image
                      text, # text
                      (bb[0], bb[1]-5), # position
                      cv2.FONT_HERSHEY_SIMPLEX, # font
                      3, # font scale
                      (0, 0, 0), # color
                      line_width, # thickness
                      cv2.LINE_AA # line type
                      )
    return img

def save_ROI(result: Results, idx: np.ndarray, save_name: str, line_width: int = 3) -> None:
    image = result.orig_img
    names = [v for _, v in result.names.items()]
    idz = np.where(np.isin(result.boxes.cls.numpy(), idx))[0].tolist()
    img = image.copy()
    for kn, id in enumerate(idz):
        bb = result.boxes.xyxy[id].data.numpy().astype(int)
        poly = result.masks.xy[id].astype(int)
        args = (kn, idx[0], bb, poly, img, names, line_width)
        img = insert_dets(args)

    cv2.imwrite(save_name, img)
    return None

def save_ROI_parallel(result: Results, idx: np.ndarray, save_name: str, line_width: int = 3) -> None:
    image = result.orig_img
    names = [v for _, v in result.names.items()]
    idz = np.where(np.isin(result.boxes.cls.numpy(), idx))[0].tolist()
    img = image.copy()

    def process_dets(kn, id):
        bb = result.boxes.xyxy[id].data.numpy().astype(int)
        poly = result.masks.xy[id].astype(int)
        args = (kn, idx[0], bb, poly, img, names, line_width)
        return insert_dets(args)
    
    # use ThreadPoolExecutor to parallelize the saving of the images
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dets, kn, id) for kn, id in enumerate(idz)]
        for future in futures:
            img = future.result()
        
    cv2.imwrite(save_name, img)
    return None
    
def region_properties(poly: np.ndarray) -> pd.DataFrame:
    cnt = np.array(poly)
    properties = {}

    # Area and Perimeter
    properties['RP_Area'] = cv2.contourArea(cnt, oriented=False)
    properties['RP_Perimeter'] = cv2.arcLength(cnt, True)

    # Bounding box properties
    x, y, w, h = cv2.boundingRect(cnt)
    properties.update({
        'RP_BB_x': x,
        'RP_BB_y': y,
        'RP_BB_w': w,
        'RP_BB_h': h
    })

    # Hu moments
    M = cv2.moments(cnt)
    Hu_Moments = cv2.HuMoments(M)
    for i in range(7):
        properties[f'RP_Hu_{i}'] = Hu_Moments[i][0]

    # Aspect ratio, Extent, Solidity, Equivalent diameter
    properties['RP_Aspect_ratio'] = w / h
    hull = cv2.convexHull(cnt)
    properties['RP_Extent'] = properties['RP_Area'] / (w * h)
    properties['RP_Solidity'] = properties['RP_Area'] / cv2.contourArea(hull)
    properties['RP_Equivalent_diameter'] = np.sqrt(4 * properties['RP_Area'] / np.pi)

    # Ellipse properties
    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    properties.update({
        'RP_Orientation': angle,
        'RP_Major_axis_length': MA,
        'RP_Minor_axis_length': ma
    })

    # Feret diameter
    _, (FW, FH), _ = cv2.minAreaRect(cnt)
    properties.update({
        'RP_Feret_diameter_max': FW,
        'RP_Feret_diameter_min': FH
    })

    return pd.DataFrame(properties, index=[0])

def GLCM_feats(image: np.ndarray,
               prefix: str = 'Gray') -> pd.DataFrame:
    feat_glcm = {}
    GLCM_0 = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=2**8)
    feat_glcm[prefix + "_GLCM_contrast"] = graycoprops(GLCM_0, 'contrast').mean()
    feat_glcm[prefix + "_GLCM_dissimilarity"] = graycoprops(GLCM_0, 'dissimilarity').mean()
    feat_glcm[prefix + "_GLCM_correlation"] = graycoprops(GLCM_0, 'correlation').mean()
    feat_glcm[prefix + "_GLCM_homogeneity"] = graycoprops(GLCM_0, 'homogeneity').mean()
    feat_glcm[prefix + "_GLCM_energy"] = graycoprops(GLCM_0, 'energy').mean()
    feat_glcm[prefix + "_GLCM_ASM"] = graycoprops(GLCM_0, 'ASM').mean()

    return pd.DataFrame(feat_glcm, index=[0])

def color_feats(image: np.ndarray,
                prefix: str = 'Gray') -> pd.DataFrame:
    feat_color = {}
    non_zero_image = image[image != 0]  # Consider only non-zero values in the image
    feat_color[prefix + '_Color_Mean'] = np.mean(non_zero_image)
    feat_color[prefix + '_Color_Median'] = np.median(non_zero_image)
    feat_color[prefix + '_Color_Std'] = np.std(non_zero_image)
    feat_color[prefix + '_Color_Skew'] = skew(non_zero_image.reshape(-1))
    feat_color[prefix + '_Color_Kurtosis'] = kurtosis(non_zero_image.reshape(-1))

    return pd.DataFrame(feat_color, index=[0])

def LBP_feats(image: np.ndarray,
              P: int = 8, # number of points
              R: int = 1, # radius
              bins: int = 16, # number of bins
              prefix: str = 'Gray') -> pd.DataFrame:
    feat_lbp = {}
    lbp = local_binary_pattern(image, P=P, R=R)
    lbp_0, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
    for i in range(len(lbp_0)):
        feat_lbp[prefix + f'_LBP_{i}'] = lbp_0[i]

    return pd.DataFrame(feat_lbp, index=[0])

def entropyplus(image: np.ndarray) -> Tuple[float, float, float]:  
    histogram         = np.histogram(image, bins=2**8, range=(0,(2**8)-1), density=True)
    histogram_prob    = histogram[0]/sum(histogram[0])    
    single_entropy    = np.zeros((len(histogram_prob)), dtype = float)
    for i in range(len(histogram_prob)):
        if(histogram_prob[i] == 0):
            single_entropy[i] = 0
        else:
            single_entropy[i] = histogram_prob[i]*np.log2(histogram_prob[i]);
    smoothness   = 1- 1/(1 + np.var(image/2**8))            
    uniformity   = sum(histogram_prob**2);        
    entropy      = -(histogram_prob*single_entropy).sum()
    return smoothness, uniformity, entropy

def Entropy_feats(image: np.ndarray,
                    prefix: str = 'Gray') -> pd.DataFrame:
    feat_entropy = {}
    smoothness, uniformity, entropy = entropyplus(image)
    feat_entropy[prefix + '_Entropy_Smoothness'] = smoothness
    feat_entropy[prefix + '_Entropy_Uniformity'] = uniformity
    feat_entropy[prefix + '_Entropy_Entropy'] = entropy

    return pd.DataFrame(feat_entropy, index=[0])

def process_image(img: np.ndarray,
                  poly: np.ndarray, 
                  prefixes: list) -> pd.DataFrame:
    other_feats = pd.DataFrame()
    for i in range(img.shape[2]):
        prefix = prefixes[i]
        image = img[:,:,i]
        entropy = Entropy_feats(image, prefix=prefix)
        lbp = LBP_feats(image, prefix=prefix)
        color = color_feats(image, prefix=prefix)
        glcm = GLCM_feats(image, prefix=prefix)
        other_feats = pd.concat([other_feats, entropy, lbp, color, glcm], axis=1)
    RP = region_properties([poly])
    return pd.concat([RP, other_feats], axis=1)

def get_all_features(results: Results,
                        name: str = 'Hops'
                        ) -> pd.DataFrame:
    All_feat = pd.DataFrame()
    ids = get_ids(results, name)
    Imgs, Polys = extract_ROI(results, ids)
    prefixes = ['Blue', 'Green', 'Red']

    pbar = tqdm(Imgs.items(), total=len(Imgs), desc='Extracting features')
    for key, img in Imgs.items():
        poly = Polys[key]
        poly = np.array(poly)
        
        All_feat = pd.concat([All_feat, 
                              process_image(img, poly, prefixes)], 
                              axis=0, ignore_index=True)
        
        pbar.update(1)
    pbar.close()
    return All_feat

def get_all_features_parallel(results: Results,
                        name: str = 'Hops'
                        ) -> pd.DataFrame:
    All_feat = pd.DataFrame()
    ids = get_ids(results, name)
    Imgs, Polys = extract_ROI(results, ids)
    prefixes = ['Blue', 'Green', 'Red']

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, img, poly, prefixes) for img, poly in zip(Imgs.values(), Polys.values())]
        for future in tqdm(futures, total=len(futures), desc='Extracting features'):
            All_feat = pd.concat([All_feat, future.result()], axis=0, ignore_index=True)

    return All_feat
