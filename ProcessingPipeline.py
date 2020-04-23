import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, glob
import random
import pickle
from CamCal import caliberation_data
# import cv2
from cv2 import cv2
from collections import deque
from typing import Deque, List, Dict
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
# try:
#     from cv2 import cv2
# except:
#     import cv2
import ProcessImage

SRC = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
DST = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, img_size):
        self.max_que_len = 10
        self.img_size = img_size
        # was the line detected in the last iteration?
        self.weight = np.array(list(range(1, self.max_que_len + 1)))
        self.weight = self.weight / np.sum(self.weight)
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted: Deque = deque()
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        self.besty = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.diffs = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def calc_radius(self, x_pts, y_pts):
        '''
        Calculate the radius in meters
        :return: float
        '''
        ym_per_pix = 3.048 / 100
        xm_per_pix = 3.7 / 378
        radius = 0
        h = self.img_size[0]
        ploty = np.linspace(0, h - 1, h)
        y_eval = np.max(ploty)
        if len(x_pts) > 0:
            new_fit = np.polyfit(y_pts * ym_per_pix, x_pts * xm_per_pix, 2)
            radius = ((1 + (2 * new_fit[0] * y_eval * ym_per_pix + new_fit[1]) ** 2) ** 1.5) / np.absolute(
                2 * new_fit[0])
        return radius

    def add_fitted(self, fit):
        '''
        Adds new fitted model to the deque, and maintain relative size.
        compute mean of all past N models as best model for a given frame
        :param fit: Current Fit
        :return: Best fit
        '''
        # print(50 * '=')
        # print('Current_fit = ', fit)
        # print('recent_fitted = ', self.recent_xfitted)
        if len(self.recent_xfitted) > self.max_que_len:
            self.recent_xfitted.popleft()
        self.recent_xfitted.append(fit)
        best_fit = np.mean(self.recent_xfitted, axis=0)

        # print('best fit = ', best_fit)
        # print(50 * '=')
        return best_fit

    def is_cur_validity(self, fit):
        '''
        To maintain stability for initial few frames
        :param fit: model fit
        :return:boolean
        '''
        if len(self.recent_xfitted) < 4 or len(self.recent_xfitted) >= self.max_que_len - 1:
            return True
        best_fit = np.mean(self.recent_xfitted, axis=0)

        if abs(fit[0]) > 3 * abs(best_fit[0]) or abs(fit[1]) > 3 * abs(best_fit[1]) or abs(fit[2]) > 3 * abs(
                best_fit[2]):
            return False

        return True

    def add_points(self, x, y):
        '''
        Add points from image to the lines, These points are activated pixel points on the image. which are added
        by the main method. After adding these points this method compute the best fit based on those points
        :param x: ndarray
        :param y:ndarray
        :return: best fit
        '''
        if False and self.bestx is not None and self.besty is not None:
            outlier_rem = [abs(x - np.mean(self.bestx)) < 3 * np.std(self.bestx)]
            x = x[outlier_rem]
            y = y[outlier_rem]
        self.allx = x
        self.ally = y
        if len(self.allx) > 3 and len(self.ally) > 3:
            self.current_fit = self._compute_cur_fit(self.allx, self.ally)
            if self.is_cur_validity(self.current_fit):
                self.best_fit = self.add_fitted(self.current_fit)

        self.bestx, self.besty = self._calcute_inds(self.best_fit)
        self.radius_of_curvature = self.calc_radius(self.bestx, self.besty)
        return self.bestx, self.besty

    def _compute_cur_fit(self, x: np.ndarray, y: np.ndarray):
        # print(list(x))
        # print(list(y))

        fit = np.polyfit(y, x, 2)

        # x = x.reshape((-1, 1))
        # poly = PolynomialFeatures(2, include_bias=False)
        # z = poly.fit_transform(x)
        # las = Lasso(alpha=1)
        # model = las.fit(z, y)
        # fit = [0 for _ in range(3)]
        # fit[0] = model.coef_[1]
        # fit[1] = model.coef_[0]
        # fit[2] = model.intercept_
        return fit

    def _calcute_inds(self, fit):
        y_size = self.img_size[0]
        out_y = np.linspace(0, y_size - 1, y_size)
        out_y = out_y.astype(float)
        try:
            out_x = fit[0] * (out_y ** 2) + fit[1] * out_y + fit[2]
        except:
            out_x = 1 * (out_y ** 2) + out_y

        out_x[out_x >= self.img_size[1]] = self.img_size[1] - 1
        out_x[out_x < 0] = 0

        out_x = out_x.astype(int)
        out_y = out_y.astype(int)
        return out_x, out_y


class ProcessingPipeline:
    def __init__(self):
        self.frame_cntr = 1
        self.mtx, self.dist = caliberation_data()
        self.img_size = (720, 1280, 3)
        img_size = (720, 1280, 3)
        self.src = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
        self.dst = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])
        # self.src = np.float32(
        #     [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        #      [((img_size[0] / 6) - 10), img_size[1]],
        #      [(img_size[0] * 5 / 6) + 60, img_size[1]],
        #      [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        # self.dst = np.float32(
        #     [[(img_size[0] / 4), 0],
        #      [(img_size[0] / 4), img_size[1]],
        #      [(img_size[0] * 3 / 4), img_size[1]],
        #      [(img_size[0] * 3 / 4), 0]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.MInv = cv2.getPerspectiveTransform(self.dst, self.src)
        self.imgProc = ProcessImage.ProcessImage()
        self.left_coeff = None
        self.right_coeff = None
        self.left_line = Line(self.img_size)
        self.right_line = Line(self.img_size)

    def get_undist_img_test(self, img):

        undist = self.undistort_image(img)
        pres = self.get_birds_eye(undist)
        # proc_img = self.imgProc.pipeline(pres)
        return pres

    def _sharpen_image(self, image):
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 3, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        return sharpened

    def undistort_image(self, image, debug=False):
        '''
        Undistort an image
        :param img:
        :return: undistorted image
        '''
        dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        if debug:
            plt.figure(figsize=(15, 10))
            plt.imshow(dst)
        return dst

    def get_birds_eye(self, image, display=False):
        '''
        Compute the perspective transformation of the image
        :param image: ndarray
        :param display: boolean debugging
        :return:warped ndarray
        '''
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self.M, (w, h), flags=cv2.INTER_LINEAR)
        if display:
            self.plot_side_by_side(image, warped)
        return warped

    def plot_side_by_side(self, img, undist, name1='Before', name2='After', save=False, num=1, fpath=''):
        '''
        helper to display image
        '''
        cmap_img = None
        cmap_undist = None
        if len(img.shape) == 2:
            cmap_img = 'gray'
        if len(undist.shape) == 2:
            cmap_undist = 'gray'
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(img, cmap=cmap_img)
        ax1.set_title(name1, fontsize=10)
        ax2.imshow(undist, cmap=cmap_undist)
        ax2.set_title(name2, fontsize=10)
        if save:
            name = f'{fpath}/processed_{num}.png'
            plt.savefig(name)
        plt.show()

    def draw_result(self, clean_image, warped_img, x_left, y_left, x_right, y_right):
        '''
        Over lay lane lines on top of original images
        :return: ndarray (overlayed image)
        '''
        warp_zero = np.zeros_like(warped_img, dtype=np.uint8)
        color_zero = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([x_left, y_left]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([x_right, y_right])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_zero, np.int_([pts]), (0, 255, 0))
        new_warp = cv2.warpPerspective(color_zero, self.MInv, (clean_image.shape[1], clean_image.shape[0]))
        result = cv2.addWeighted(clean_image, 1, new_warp, 0.3, 0)
        return result

    def find_line_around(self, bin_warped: np.ndarray, fit_params: List, margin_fit=60):
        '''
        search for lane lines in the current frame using model from previous frame.
        Use the model to find x points in the current and the using window acoss those points find activated pixel
        :param margin_fit: margin to search for
        :return: activated points
        '''
        # if self.frame_cntr > 6:
        #     plt.imshow(bin_warped)
        #     plt.show()
        non_zeros = bin_warped.nonzero()
        y_non_zeros = np.array(non_zeros[0])
        x_non_zeros = np.array(non_zeros[1])
        fitted_mid = (fit_params[0] * (y_non_zeros ** 2)) + (fit_params[1] * y_non_zeros) + fit_params[2]
        lane_inds = ((x_non_zeros > (fitted_mid - margin_fit)) & (x_non_zeros < (fitted_mid + margin_fit)))
        x_pts = x_non_zeros[lane_inds]
        y_pts = y_non_zeros[lane_inds]
        return x_pts, y_pts

    def process(self, img, debug=False):
        undist = self.undistort_image(img)
        pres = self.get_birds_eye(undist)
        proc_img = self.imgProc.pipeline(pres)
        out_img = None
        # cv2.rectangle(proc_img, (20, 20), (90, 90), (0, 255, 255), 2)
        # plt.imshow(proc_img)
        # plt.show()
        # if self.frame_cntr >= 7:
        #     print("Frame 6")

        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            left_x, left_y = self.find_line_around(proc_img, self.left_line.best_fit)
            right_x, right_y = self.find_line_around(proc_img, self.right_line.best_fit)

            if len(left_x) > 50 and len(right_x) > 50:
                x_left_fitted, y_left_fitted = self.left_line.add_points(left_x, left_y)
                x_right_fitted, y_right_fitted = self.right_line.add_points(right_x, right_y)
            else:
                left_x, left_y, right_x, right_y, out_img = self.find_lane_pixel_window(proc_img)
                x_left_fitted, y_left_fitted = self.left_line.add_points(left_x, left_y)
                x_right_fitted, y_right_fitted = self.right_line.add_points(right_x, right_y)
        else:
            left_x, left_y, right_x, right_y, out_img = self.find_lane_pixel_window(proc_img)
            x_left_fitted, y_left_fitted = self.left_line.add_points(left_x, left_y)
            x_right_fitted, y_right_fitted = self.right_line.add_points(right_x, right_y)
            # x_left_fitted, y_left_fitted = self.left_line.compute_best_fit()
            # x_right_fitted, y_right_fitted = self.right_line.compute_best_fit()

        # x_left_fitted, y_left_fitted = self.fit_poly(left_x, left_y)
        # x_right_fitted, y_right_fitted = self.fit_poly(right_x, right_y)
        y_left_fitted = y_left_fitted.astype(int)
        x_left_fitted = x_left_fitted.astype(int)
        x_right_fitted = x_right_fitted.astype(int)
        y_right_fitted = y_right_fitted.astype(int)
        debug = False
        if debug:
            if out_img is None:
                out_img = np.copy(proc_img)
                out_img = np.dstack((out_img, out_img, out_img))

            out_img[y_left_fitted, x_left_fitted] = (255, 0, 0)
            out_img[y_right_fitted, x_right_fitted] = (255, 0, 0)
            plt.imshow(out_img)
            plt.imsave(f'deb_out/frame_{self.frame_cntr}.png', out_img)
            plt.show()
        result = self.draw_result(undist, proc_img, x_left_fitted, y_left_fitted, x_right_fitted, y_right_fitted)
        cv2.putText(result, "L. Curvature: %.2f km" % (self.left_line.radius_of_curvature / 1000), (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(result, "R. Curvature: %.2f km" % (self.right_line.radius_of_curvature / 1000), (50, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 255, 255), 2)


        center = ((self.img_size[1]//2) - ((x_right_fitted[0] + x_left_fitted[0]) / 2)) * 3.7 / 700

        cv2.putText(result, "C. Position: %.2f m" % (
            center), (50, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        self.frame_cntr += 1
        return result

    def _get_lane_hist(self, img):
        '''
        Compute histogram of the activated pixels
        :param img: ndarray
        '''
        histogram = np.sum(img[img.shape[0] // 3:, :], axis=0)
        return histogram

    def un_warp_image(self, image):
        '''
        Inverse perspective transformation to get back to original image
        :return: unwarped images
        '''
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self.MInv, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def find_lane_pixel_window(self, image: np.ndarray, n_window=9, margin=100, minpx=50, debug=False):
        '''
        N window sliding method to find activated pixels for lane lines
        :return:  activated left and right points
        '''
        out_img = np.copy(image)
        # out_img = out_img.astype(int)
        out_img = np.dstack((out_img, out_img, out_img))
        hist = self._get_lane_hist(image)
        mid = np.int(hist.shape[0] // 2)
        left_base = mid // 2
        if np.argmax(hist[:mid]) > 0:
            left_base = np.argmax(hist[:mid])
        right_base = mid + mid // 2
        if np.argmax(hist[mid:]) > 0:
            right_base = np.argmax(hist[mid:]) + mid
        win_height = image.shape[0] // n_window
        non_zero = image.nonzero()
        y_non_zero = non_zero[0]
        x_non_zero = non_zero[1]
        if len(y_non_zero) > 3 and len(x_non_zero) > 3:

            inds_good_left = []
            inds_good_right = []
            left_x_cur = left_base
            right_x_cur = right_base
            for win in range(n_window):
                win_y_low = int(image.shape[0] - (win + 1) * win_height)
                win_y_high = int(image.shape[0] - (win) * win_height)

                win_xleft_low = int(left_x_cur - margin)
                win_xleft_high = int(left_x_cur + margin)
                win_xright_low = int(right_x_cur - margin)
                win_xright_high = int(right_x_cur + margin)

                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)

                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

                # plt.imshow(out_img)
                # plt.show()

                left_inds = ((y_non_zero >= win_y_low) & (y_non_zero < win_y_high) &
                             (x_non_zero >= win_xleft_low) & (x_non_zero < win_xleft_high)).nonzero()[0]
                right_inds = ((y_non_zero >= win_y_low) & (y_non_zero < win_y_high) &
                              (x_non_zero >= win_xright_low) & (x_non_zero < win_xright_high)).nonzero()[0]

                inds_good_left.append(left_inds)
                inds_good_right.append(right_inds)
                if len(left_inds) > minpx:
                    left_x_cur = np.int(np.mean(x_non_zero[left_inds]))
                if len(right_inds) > minpx:
                    right_x_cur = np.int(np.mean(x_non_zero[right_inds]))

            try:
                inds_good_left = np.concatenate(inds_good_left)
                inds_good_right = np.concatenate(inds_good_right)
            except:
                pass
            leftx = x_non_zero[inds_good_left]
            lefty = y_non_zero[inds_good_left]
            rightx = x_non_zero[inds_good_right]
            righty = y_non_zero[inds_good_right]
        else:
            # leftx, lefty, rightx, righty = np.array(),np.array(),np.array(),np.array()
            leftx = np.repeat(left_base, 35)
            rightx = np.repeat(right_base, 35)
            lefty = np.array(range(image.shape[0] - 1, image.shape[0] - 36, -1))
            righty = np.array(range(image.shape[0] - 1, image.shape[0] - 36, -1))

        debug = False
        if debug:
            out_img[lefty, leftx] = (255, 0, 0)
            out_img[righty, rightx] = (255, 0, 0)
            plt.imshow(out_img)
            plt.show()

        return leftx, lefty, rightx, righty, out_img


def test_prespective_tx():
    print()
    img = plt.imread('test_images/test2.jpg')
    # plt.imshow(img)
    # plt.show()
    proc = ProcessingPipeline()
    undist = proc.undistort_image(img)
    pts = np.array(proc.src, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 255), thickness=5)
    pres = proc.get_birds_eye(img)
    unpres = proc.un_warp_image(pres)
    proc.plot_side_by_side(img, pres)
    proc.plot_side_by_side(unpres, pres)


def test_video(inputVideo, outputVideo):
    myclip = VideoFileClip(inputVideo)
    proc = ProcessingPipeline()

    def pipe(image):
        return proc.process(image)

    clip = myclip.fl_image(pipe)
    clip.write_videofile(outputVideo, audio=False)

    print()


def test_img():
    img = plt.imread('test_images/test1.jpg')
    img2 = plt.imread('test_images/test2.jpg')
    img3 = plt.imread('test_images/test3.jpg')
    # # plt.imshow(img)
    # # plt.show()
    proc = ProcessingPipeline()
    op = proc.process(img)
    op2 = proc.process(img2)
    op3 = proc.process(img3)
    proc.plot_side_by_side(img, op)
    proc.plot_side_by_side(img, op2)
    proc.plot_side_by_side(img, op3)


if __name__ == '__main__':
    print()
    # test_prespective_tx()
    # test_video('test_vids/harder_challenge_video.mp4', 'test_vids/out_harder_challenge_video.mp4')
    # test_video('test_vids/project_video.mp4', 'test_vids/out_project_video.mp4')
    # test_video('test_vids/challenge_video.mp4', 'test_vids/out_challenge_video.mp4')
    # test_video('test_vids/challenge_short.mp4', 'test_vids/out_challenge_short.mp4')
    # test_img()


def test_prespective_tx():
    print()
    img = plt.imread('test_images/test5.jpg')
    # plt.imshow(img)
    # plt.show()
    proc = ProcessingPipeline()
    undist = proc.undistort_image(img)
    pts = np.array(proc.src, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 0, 255), thickness=5)
    pres = proc.get_birds_eye(img)
    unpres = proc.un_warp_image(pres)
    proc.plot_side_by_side(img, pres)
    proc.plot_side_by_side(unpres, pres)

# if self.left_line.bestx is not None and self.right_line.bestx is not None:
# outlier_rem = [abs(left_x - np.mean(self.left_line.bestx)) < 3 * np.std(self.left_line.bestx)]
# left_x = left_x[outlier_rem]
# left_y = left_y[outlier_rem]
# outlier_rem = [abs(right_x - np.mean(self.right_line.bestx)) < 3 * np.std(self.right_line.bestx)]
# right_x = right_x[outlier_rem]
# right_y = right_y[outlier_rem]
