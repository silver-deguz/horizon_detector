import numpy as np
import cv2
import pickle
from pathlib import Path 
from sklearn import linear_model
import tqdm
import time 
import matplotlib.pyplot as plt


def detect_horizon(input_dir, output_dir=None, plot_gt=True):

    images = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.png')) + list(Path(input_dir).glob('*.jpeg'))

    xl, xr = 0, 1919

    for ix, frame in enumerate(images): 
        
        fig, ax = plt.subplots(figsize=(18,26), nrows=1, ncols=2)

        bgr = cv2.imread(str(frame))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 180, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 80)

        yl_avg, yr_avg = 0, 0
        for ix, r_theta in enumerate(lines):
            r, theta = r_theta[0] 
            yl = (-1 / np.tan(theta)) * xl + (r / np.sin(theta))
            yr = (-1 / np.tan(theta)) * xr + (r / np.sin(theta))

            yl_avg += yl 
            yr_avg += yr 

        if len(lines) > 0:
            yl_avg = yl_avg // len(lines)
            yr_avg = yr_avg // len(lines)
        ax[1].plot([xl, xr], [yl_avg, yr_avg], '-r') # averaged
        ax[0].imshow(edges, cmap='gray')
        ax[1].imshow(gray, cmap='gray')

        plt.show()


class HorizonLineDetection():
    def __init__(self, gaussian_params_file=None):
        self.scale_percent = 1/2

        if gaussian_params_file:
            with open(gaussian_params_file, 'rb') as f:
                self.mu = pickle.load(f)
                self.covar = pickle.load(f)
                self.a = pickle.load(f)

    '''
    Segments image into sky and nonsky regions by running gaussian color
    classifier model on each pixel.
    '''
    def segment_image(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        X = hsv_img.reshape((img.shape[0]*img.shape[1]), img.shape[2])

        # classify pixels as sky or non-sky
        probs = np.zeros((len(X),2))

        diff0 = X - self.mu[0]
        diff1 = X - self.mu[1]
        diff0T = np.transpose(diff0)
        diff1T = np.transpose(diff1)

        inv_cov0 = np.linalg.inv(self.covar[0])
        inv_cov1 = np.linalg.inv(self.covar[1])

        for i in range(len(X)):
            probs[i,0] = (-0.5 * (diff0[i,:] @ inv_cov0 @ diff0T[:,i])) + self.a[0]
            probs[i,1] = (-0.5 * (diff1[i,:] @ inv_cov1 @ diff1T[:,i])) + self.a[1]

        y_hat = np.argmax(probs, axis=1)
        y_hat[y_hat != 0] = -1
        y_hat += 1

        # Create binary mask
        y_mask = y_hat.reshape((img.shape[0], img.shape[1]))
        y_mask = y_mask.astype(np.uint8)*255
        return y_mask
    
    '''
    Post-processing to remove noise and fill gaps in segmented image.
    Applies simple morphological operations.
    '''
    def postprocess(self, img):
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
        mask = img.copy()
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=kernel1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel2, iterations=3)
        inverted_mask = cv2.bitwise_not(mask)
        return mask, inverted_mask 
    
    '''
    Returns predicted horizon line which in form (x1, y1, x2, y2) where
    x1, y1 is the left endpoint and x2, y2 is the right endpoint
    '''
    def get_line_prediction(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_TREE,\
											cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        cnt = contours[0]
        X, y = cnt[:,:,0], cnt[:,:,1]
  
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X.reshape(-1, 1), y)

        x1, x2 = 0, img.shape[1] - 1    
        assert x2 > x1

        line_X = np.arange(x1, x2)[:, np.newaxis]
        line_y = ransac.predict(line_X)
        y1, y2 = line_y[0], line_y[-1]
        line_pred = (x1, int(y1[0]), x2, int(y2[0]))
        return line_pred

    '''
    Given line_pred in form (x1, y1, x2, y2) and img,
    draw line on the image,
    '''
    @staticmethod
    def draw(img, line_pred):
        x1, y1, x2, y2 = line_pred
        left_pt = (x1, y1)
        right_pt = (x2, y2)
        img = cv2.line(img, left_pt, right_pt, color=(100, 0, 180), thickness=4)
        return img

    '''
    Runs horizon line detection on an image
    '''
    def detect(self, img):
        w, h = img.shape[1], img.shape[0]
        bgr = img.copy()

        # resize image to speed up segmentation
        resize_w = int(w * self.scale_percent)
        resize_h = int(h * self.scale_percent)
        bgr_scaled = cv2.resize(img, (resize_w,resize_h))

        # segment image using gaussian color model predictions
        binary_img = self.segment_image(bgr_scaled)

        # apply postprocessing to clean up image
        mask, inv_mask = self.postprocess(binary_img)

        # upsize the image back
        segment_img = cv2.resize(mask, (w,h))
        inv_segment_img = cv2.resize(inv_mask, (w,h))

        # get line endpoints
        line_pred = self.get_line_prediction(segment_img)
        inv_line_pred = self.get_line_prediction(inv_segment_img)

        return line_pred, inv_line_pred
        

def main(input_dir, output_dir, show_evaluation=False):
    
    images = sorted(list(Path(input_dir).glob('*.jpg')))

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # get predictions
    print("Detecting")
    hld = HorizonLineDetection()
    
    pred = dict()

    tic = time.time()

    for ix, frame in tqdm(enumerate(images)):
     
        bgr = cv2.imread(str(frame))
        line_pred, _ = hld.detect(bgr)

        pred[frame.name] = line_pred

        res = hld.draw(bgr, line_pred)

        out_path = Path(f'{output_dir}/{frame.name}')
        cv2.imwrite(str(out_path), res)

    toc = time.time() - tic
    print(f'Time elapsed: {toc}s')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="path to input directory of images")
    parser.add_argument("--output", required=True, type=str, help="path to output director to save results to")
    parser.add_argument("--evaluate", required=False, help="run evaluation and print metrics to console", action="store_true")
    args = parser.parse_args()

    main(args.input, args.output, args.evaluate)

    
