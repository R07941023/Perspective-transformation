import numpy as np
import cv2
import sys

def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None
    # matrix
    A = np.zeros((2*N, 8))
    A[0][0], A[0][1], A[0][2], A[0][3], A[0][4], A[0][5], A[0][6], A[0][7] = u[0][0], u[0][1], 1, 0, 0, 0, - u[0][0]*v[0][0], - u[0][1]*v[0][0]
    A[1][0], A[1][1], A[1][2], A[1][3], A[1][4], A[1][5], A[1][6], A[1][7] = 0, 0, 0, u[0][0], u[0][1], 1, - u[0][0]*v[0][1], - u[0][1]*v[0][1]
    A[2][0], A[2][1], A[2][2], A[2][3], A[2][4], A[2][5], A[2][6], A[2][7] = u[1][0], u[1][1], 1, 0, 0, 0, - u[1][0]*v[1][0], - u[1][1]*v[1][0]
    A[3][0], A[3][1], A[3][2], A[3][3], A[3][4], A[3][5], A[3][6], A[3][7] = 0, 0, 0, u[1][0], u[1][1], 1, - u[1][0]*v[1][1], - u[1][1]*v[1][1]
    A[4][0], A[4][1], A[4][2], A[4][3], A[4][4], A[4][5], A[4][6], A[4][7] = u[2][0], u[2][1], 1, 0, 0, 0, - u[2][0]*v[2][0], - u[2][1]*v[2][0]
    A[5][0], A[5][1], A[5][2], A[5][3], A[5][4], A[5][5], A[5][6], A[5][7] = 0, 0, 0, u[2][0], u[2][1], 1, - u[2][0]*v[2][1], - u[2][1]*v[2][1]
    A[6][0], A[6][1], A[6][2], A[6][3], A[6][4], A[6][5], A[6][6], A[6][7] = u[3][0], u[3][1], 1, 0, 0, 0, - u[3][0]*v[3][0], - u[3][1]*v[3][0]
    A[7][0], A[7][1], A[7][2], A[7][3], A[7][4], A[7][5], A[7][6], A[7][7] = 0, 0, 0, u[3][0], u[3][1], 1, - u[3][0]*v[3][1], - u[3][1]*v[3][1]
    b = v.reshape(( 2 * N, 1) )
    H = np.dot(np.linalg.inv(A), b)
    H = np.append(H, [1]).reshape(3, 3)
    return H

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = canvas.shape
    canvas_corners = np.array([[0, 0], [canvas.shape[0], 0], [0, canvas.shape[0]], [canvas.shape[0], canvas.shape[0]]] )
    # transform matrix
    warp = solve_homography( canvas_corners.astype( np.float32 ), corners.astype( np.float32 ) )
    # resize
    img = cv2.resize( img, (h, h), cv2.INTER_CUBIC )
    # warpPerspective
    img_warp = cv2.warpPerspective( img, warp, (w, h) )
    # mapping
    mask = cv2.cvtColor( img_warp, cv2.COLOR_BGR2GRAY )
    mask[mask != 0] = 255
    mask = 255-mask
    canvas = cv2.bitwise_and( canvas, canvas, mask=mask )
    canvas = canvas+img_warp
    return canvas


def findpoint(kp1, matches):
    list_kp1 = np.array( [kp1[mat.queryIdx].pt for mat in matches] )
    x = list_kp1[:, 0]
    y = list_kp1[:, 1]
    point1 = list_kp1[np.argwhere(x==np.min(x))[0][0]]
    point2 = list_kp1[np.argwhere(y==np.min(y))[0][0]]
    point3 = list_kp1[np.argwhere( y == np.max( y ) )[0][0]]
    point4 =  list_kp1[np.argwhere( x == np.max( x ) )[0][0]]
    canvas_corners = np.array( [point1, point2, point3, point4] )
    return canvas_corners

def brightness_threshold(img):
    threshold = (10, 100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres_img = cv2.inRange( gray, threshold[0], threshold[1] )
    # morphology
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (7, 7) )
    morphology_img = cv2.morphologyEx(thres_img, cv2.MORPH_OPEN, kernel)
    morphology_img = cv2.morphologyEx( morphology_img, cv2.MORPH_CLOSE, kernel )
    # zero mapping
    rang = [[300, 1350], [100, 1075]]
    zero_mapping = morphology_img.copy()
    zero_mapping[0:rang[1][0], :] = 0
    zero_mapping[rang[1][1]:, :] = 0
    zero_mapping[:, :rang[0][0]] = 0
    zero_mapping[:, rang[0][1]:] = 0

    # zero_mapping = cv2.cvtColor(zero_mapping, cv2.COLOR_GRAY2BGR)
    # cv2.line( zero_mapping, tuple( [rang[0][0], rang[1][0]] ), tuple( [rang[0][1], rang[1][0]] ), (255, 0, 0), 5 )
    # cv2.line( zero_mapping, tuple( [rang[0][1], rang[1][0]] ), tuple( [rang[0][1], rang[1][1]] ), (255, 0, 0), 5 )
    # cv2.line( zero_mapping, tuple( [rang[0][1], rang[1][1]] ), tuple( [rang[0][0], rang[1][1]] ), (255, 0, 0), 5 )
    # cv2.line( zero_mapping, tuple( [rang[0][0], rang[1][1]] ), tuple( [rang[0][0], rang[1][0]] ), (255, 0, 0), 5 )
    # zero_mapping = cv2.resize( zero_mapping, (500, 500), cv2.INTER_CUBIC )
    # cv2.imshow( 'zero_mapping', zero_mapping )
    # cv2.waitKey(10)
    return zero_mapping

def findpoint_from_dark(dark_img):
    # print(dark_img)
    index = np.argwhere( dark_img == 255 )
    index[:, [0, 1]] = index[:, [1, 0]]
    if index.shape[0] < 4:
        point1 = np.array([0, 0])
        point2 = np.array([0, 0])
        point3 = np.array([0, 0])
        point4 = np.array([0, 0])
        canvas_corners = np.array( [point1, point2, point3, point4] )
        return canvas_corners
    rect = cv2.minAreaRect( index )
    box = cv2.boxPoints( rect )
    box = np.int0( box )
    point1 = box[1]
    point2 = box[2]
    point3 = box[0]
    point4 = box[3]
    canvas_corners = np.array( [point1, point2, point3, point4] )

    dark_img = cv2.cvtColor( dark_img, cv2.COLOR_GRAY2BGR )
    cv2.line( dark_img, tuple( canvas_corners[0].astype( np.int ) ), tuple( canvas_corners[1].astype( np.int ) ), (255, 0, 0), 5 )
    cv2.line( dark_img, tuple( canvas_corners[1].astype( np.int ) ), tuple( canvas_corners[3].astype( np.int ) ),
              (255, 0, 0), 5 )
    cv2.line( dark_img, tuple( canvas_corners[3].astype( np.int ) ), tuple( canvas_corners[2].astype( np.int ) ),
              (255, 0, 0), 5 )
    cv2.line( dark_img, tuple( canvas_corners[2].astype( np.int ) ), tuple( canvas_corners[0].astype( np.int ) ),
              (255, 0, 0), 5 )

    dark_img = cv2.resize( dark_img, (500, 500), cv2.INTER_CUBIC )
    cv2.imshow( 'dark_img', dark_img )
    cv2.waitKey(1)
    return canvas_corners




def main(ref_image,template,video):
    ref_image = cv2.imread(ref_image)  ## load gray if you need.  # teacher
    template = cv2.imread(template)  ## load gray if you need.  # mask
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            dark_img = brightness_threshold(frame)
            canvas_corners = findpoint_from_dark(dark_img)
            try:
                frame = transform( ref_image, frame, canvas_corners )
            except:
                pass
            videowriter.write(frame)

            # # draw line
            # cv2.line( frame, tuple( canvas_corners[0].astype( np.int ) ), tuple( canvas_corners[1].astype( np.int ) ), (0, 0, 0), 5 )
            # cv2.line( frame, tuple( canvas_corners[1].astype( np.int ) ), tuple( canvas_corners[3].astype( np.int ) ), (0, 0, 0), 5 )
            # cv2.line( frame, tuple( canvas_corners[3].astype( np.int ) ), tuple( canvas_corners[2].astype( np.int ) ), (0, 0, 0), 5 )
            # cv2.line( frame, tuple( canvas_corners[2].astype( np.int ) ), tuple( canvas_corners[0].astype( np.int ) ), (0, 0, 0), 5 )

        else:
            print('Finish!!!')
            break
        i += 1

    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)