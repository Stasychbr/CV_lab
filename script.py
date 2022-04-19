import argparse
import cv2 as cv
import numpy as np
from scipy.ndimage.morphology import binary_opening


def read_contours_from_image(path):
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
    # hack for noisy images
    # actually it wouldn't be necessary if the polygons' contours
    # remain continuous
    noise_idx = img < 255
    if np.count_nonzero(noise_idx > 0):
        img[noise_idx] = 0
        img = cv.GaussianBlur(img, (3, 3), 0.5)
    # find all contours and denoise the image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    result = np.zeros(img.shape, dtype=np.uint8)
    cv.drawContours(result, contours, -1, 255, -1, cv.LINE_4)
    result = 255 * binary_opening(result, np.ones((5, 5))).astype(np.uint8)
    contours, _ = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # smoothen remain artifacts of 'lines' noise
    approxCnts = []
    for c in contours:
        eps = 0.1 * cv.arcLength(c, True)
        approxCnts.append(cv.approxPolyDP(c, eps, True))
    return approxCnts

def parse_input_contours(path):
    # parse given contours out of txt file
    file = open(path, 'r')
    file.readline()
    contours = []
    for line in file:
        coords = [float(x) for x in line.split(',')]
        cnt = np.empty((len(coords) // 2, 2), dtype=np.float32)
        cnt[:, 0] = coords[0::2]
        cnt[:, 1] = coords[1::2]
        contours.append(cnt)
    return contours

def match_contours(contours, template):
    # constants 
    treshold = 0.2
    enlarger = 50

    # help functors
    get_rot_mat = lambda angle: np.asarray([[np.math.cos(angle), -np.math.sin(angle)], 
                                            [np.math.sin(angle), np.math.cos(angle)]], dtype=np.float32)

    def enlarge_points_num(contour):
        enlarged_c = np.empty((enlarger * (contour.shape[0] - 1) + 1, contour.shape[-1]), dtype=np.float32)
        for i in range(contour.shape[0] - 1):
            pt1 = np.ravel(contour[i])
            pt2 = np.ravel(contour[i + 1])
            enlarged_c[i * enlarger : (i + 1) * enlarger, :] = np.linspace(pt1, pt2, num=enlarger, endpoint=False)
        enlarged_c[-1, :] = np.ravel(contour[-1, :])
        return enlarged_c

    # masks for IoU calculation
    cnt_mask = np.zeros((200, 300), dtype=np.uint8)
    tmp_mask = np.zeros((200, 300), dtype=np.uint8)

    result = []
    cur_metric = np.empty(len(template))
    tmp_len = np.empty(len(template))
    # calc lenght for all template polygons to calculate scale
    for i, t in enumerate(template):
        tmp_len[i] = cv.arcLength(t, True)
    for c in contours:
        # try to match found contour with templates
        for i, t in enumerate(template):
            cur_metric[i] = cv.matchShapes(c, t, cv.CONTOURS_MATCH_I2, 0)
        # if found good enough match 
        if np.min(cur_metric) < treshold:
            # clear masks 
            cnt_mask.fill(0)
            tmp_mask.fill(0)
            # matched shape
            i_match = np.argmin(cur_metric)

            scale = cv.arcLength(c, True) / tmp_len[i_match]
            cur_tmp = scale * templates[i_match]
            
            # set more points in contours for better regression results
            enlarged_tmp = enlarge_points_num(cur_tmp)
            
            # calculate rotation angle of the contour
            tvx, tvy, *_ = cv.fitLine(enlarged_tmp, cv.DIST_L2, 0, 0.1, 0.001)
            tmp_angle = np.arccos(tvx) * np.sign(tvy)

            enlarged_c = enlarge_points_num(c)
            cvx, cvy, *_ = cv.fitLine(enlarged_c, cv.DIST_L2, 0, 0.1, 0.001)
            cnt_angle = np.arccos(cvx) * np.sign(cvy)

            delta_angle = cnt_angle - tmp_angle

            # unfortunately, it's impossible to determine what cv.fitline 
            # returned as line direction (in terms of 180 degrees rotation)
            # so we have to explicitly learn better angle, delta_angle or delta_angle + pi
            
            # use masks to draw contours and calculate IoU metric
            cv.drawContours(cnt_mask, [c], -1, 255, -1, cv.LINE_4)
            cur_tmp = np.inner(cur_tmp, get_rot_mat(delta_angle))

            Mc = cv.moments(c)
            Mt = cv.moments(cur_tmp)
            dx = Mc['m10'] / Mc['m00'] - Mt['m10'] / Mt['m00']
            dy = Mc['m01'] / Mc['m00'] - Mt['m01'] / Mt['m00']
            
            cur_tmp[:, 0] += dx
            cur_tmp[:, 1] += dy
            
            cv.drawContours(tmp_mask, [np.reshape(np.int0(cur_tmp), (-1, 1, 2))], -1, 255, -1, cv.LINE_4)
            IoU = np.count_nonzero(np.logical_and(cnt_mask, tmp_mask)) / np.count_nonzero(np.logical_or(cnt_mask, tmp_mask))
            
            # return contour to basic position to rotate around axes origin
            cur_tmp[:, 0] -= dx
            cur_tmp[:, 1] -= dy
            cur_tmp = np.inner(cur_tmp, get_rot_mat(np.pi))
            delta_angle += np.pi
            dx1 = dx
            dy1 = dy
            Mt = cv.moments(cur_tmp)
            dx = Mc['m10'] / Mc['m00'] - Mt['m10'] / Mt['m00']
            dy = Mc['m01'] / Mc['m00'] - Mt['m01'] / Mt['m00']
            
            cur_tmp[:, 0] += dx
            cur_tmp[:, 1] += dy
            tmp_mask.fill(0)
            # calculate another IoU and compare
            cv.drawContours(tmp_mask, [np.reshape(np.int0(cur_tmp), (-1, 1, 2))], -1, 255, -1, cv.LINE_4)
            if np.count_nonzero(np.logical_and(cnt_mask, tmp_mask)) / np.count_nonzero(np.logical_or(cnt_mask, tmp_mask)) < IoU:
                dx = dx1
                dy = dy1
                delta_angle -= np.pi
            # append contour and transform parameters to result list
            result.append((i_match, int(dx), int(dy), int(scale), int(delta_angle / np.pi * 180)))
    return result

# solution based on contours processing
# result of given basic test: 0.88 (IoU metric)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', type=str)
    parser.add_argument('-i', type=str)
    args = parser.parse_args()

    template_path = args.s[1:] if args.s[0] == ' ' else args.s
    img_path = args.i[1:] if args.i[0] == ' ' else args.i

    templates = parse_input_contours(template_path)
    contours = read_contours_from_image(img_path)

    res = match_contours(contours, templates)
    print(len(res))
    for r in res:
        line = ', '.join([str(n) for n in r])
        print(line)