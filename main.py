import numpy as np
import cv2
import time
import pandas as pd

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

def warpPerspective_p3( img, H, size ):
    w, h = size
    img_warp = np.zeros((h, w, 3)).astype(np.uint8)
    ux = np.linspace(0, w-1, w)
    uy = np.linspace( 0, h - 1, h )
    map_position_matrix = []
    map_value_matrix = []
    for p in uy:
        vx = np.round( (H[0][0] * ux + H[0][1]*p + H[0][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype( np.int )
        vy = np.round( (H[1][0] * ux + H[1][1]*p + H[1][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype( np.int )
        value = img[int(p), :]
        v = np.vstack((vx, vy)).T
        map_position_matrix.append(v)
        map_value_matrix.append(value)
    map_position_matrix = np.array(map_position_matrix )
    map_value_matrix = np.array(map_value_matrix)
    for i in range(map_position_matrix.shape[0]):
        for j in range(map_position_matrix.shape[1]):
            if map_position_matrix[i][j][1] < img_warp.shape[0] and map_position_matrix[i][j][1] > 0 and map_position_matrix[i][j][0] < img_warp.shape[1] and map_position_matrix[i][j][0] > 0:
                img_warp[map_position_matrix[i][j][1]][map_position_matrix[i][j][0]] = map_value_matrix[i][j]
    return img_warp

def inverse_warpPerspective( img, H, size ):
    w, h = size
    H = np.linalg.inv(H)
    img_warp = np.zeros( (h, w, 3) ).astype( np.uint8 )
    ux = np.linspace( 0, w - 1, w )
    uy = np.linspace( 0, h - 1, h )
    map_position_matrix = []
    map_value_matrix = []
    for p in uy:
        vx = np.round( (H[0][0] * ux + H[0][1] * p + H[0][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype(np.int )
        vy = np.round( (H[1][0] * ux + H[1][1] * p + H[1][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype(np.int )
        value = img[int( p ), :]
        v = np.vstack( (vx, vy) ).T
        map_position_matrix.append( v )
        map_value_matrix.append( value )
    map_position_matrix = np.array( map_position_matrix )
    map_value_matrix = np.array( map_value_matrix )
    # print( map_value_matrix.shape )
    # print( map_position_matrix.shape )
    # print( map_position_matrix[0][0] )
    # print( map_position_matrix[0][-1] )
    # print( map_position_matrix[-1][0] )
    # print( map_position_matrix[-1][-1] )
    for i in range( map_position_matrix.shape[0] ):
        for j in range( map_position_matrix.shape[1] ):
                img_warp[i][j] = img[map_position_matrix[i][j][1]][map_position_matrix[i][j][0]]
    return img_warp

def warpPerspective_p1( img, H, size ):
    w, h = size
    img_warp = np.zeros((h, w, 3)).astype(np.uint8)
    ux = np.linspace(0, w-1, w)
    uy = np.linspace( 0, h - 1, h )
    map_position_matrix = []
    map_value_matrix = []
    # print(ux.shape, uy.shape)
    for p in uy:
        vx = np.round( (H[0][0] * ux + H[0][1]*p + H[0][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype( np.int )
        vy = np.round( (H[1][0] * ux + H[1][1]*p + H[1][2]) / (H[2][0] * ux + H[2][1] * p + H[2][2]) ).astype( np.int )
        value = img[int(p), :]
        v = np.vstack((vx, vy)).T
        map_position_matrix.append(v)
        map_value_matrix.append(value)
    map_position_matrix = np.array(map_position_matrix )
    map_value_matrix = np.array(map_value_matrix)
    for i in range(map_position_matrix.shape[0]):
        for j in range(map_position_matrix.shape[1]):
            try:
                img_warp[map_position_matrix[i][j][1]][map_position_matrix[i][j][0]] = map_value_matrix[i][j]
            except:
                pass
    return img_warp

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = canvas.shape
    canvas_corners = np.array([[0, 0], [canvas.shape[0], 0], [0, canvas.shape[0]], [canvas.shape[0], canvas.shape[0]]] )
    # transform matrix
    warp = solve_homography( canvas_corners.astype( np.float32 ), corners.astype( np.float32 ) )
    # resize
    img = cv2.resize( img, (h, h), cv2.INTER_CUBIC )
    # warpPerspective
    # img_warp = inverse_warpPerspective( img, warp, (h, h) )
    img_warp = warpPerspective_p1( img, warp, (w, h) )
    # img_warp = cv2.warpPerspective( img, warp, (w, h) )
    # mapping
    index = np.argwhere(img_warp != np.array([0, 0, 0]))
    canvas[index[:, 0], index[:, 1]] = img_warp[index[:, 0], index[:, 1]]
    # cv2.imshow('123', canvas)
    # cv2.waitKey(0)
    return canvas

def hw3_1():
    canvas = cv2.imread( './input/Akihabara.jpg' )
    img1 = cv2.imread( './input/lu.jpeg' )
    img2 = cv2.imread( './input/kuo.jpg' )
    img3 = cv2.imread( './input/haung.jpg' )
    img4 = cv2.imread( './input/tsai.jpg' )
    img5 = cv2.imread( './input/han.jpg' )
    # goal
    canvas_corners1 = np.array( [[779, 312], [1014, 176], [739, 747], [978, 639]] )
    canvas_corners2 = np.array( [[1194, 496], [1537, 458], [1168, 961], [1523, 932]] )
    canvas_corners3 = np.array( [[2693, 250], [2886, 390], [2754, 1344], [2955, 1403]] )
    canvas_corners4 = np.array( [[3563, 475], [3882, 803], [3614, 921], [3921, 1158]] )
    canvas_corners5 = np.array( [[2006, 887], [2622, 900], [2008, 1349], [2640, 1357]] )

    # transform
    canvas = transform( img1, canvas, canvas_corners1 )
    canvas = transform( img2, canvas, canvas_corners2 )
    canvas = transform( img3, canvas, canvas_corners3 )
    canvas = transform( img4, canvas, canvas_corners4 )
    canvas = transform( img5, canvas, canvas_corners5 )

    # cv2.line( canvas, tuple( canvas_corners1[0].astype( np.int ) ), tuple( canvas_corners1[1].astype( np.int ) ),
    #           (255, 0, 0), 20 )
    # cv2.line( canvas, tuple( canvas_corners1[1].astype( np.int ) ), tuple( canvas_corners1[3].astype( np.int ) ),
    #           (255, 0, 0), 20 )
    # cv2.line( canvas, tuple( canvas_corners1[3].astype( np.int ) ), tuple( canvas_corners1[2].astype( np.int ) ),
    #           (255, 0, 0), 20 )
    # cv2.line( canvas, tuple( canvas_corners1[2].astype( np.int ) ), tuple( canvas_corners1[0].astype( np.int ) ),
    #           (255, 0, 0), 20 )

    # save
    cv2.imwrite( 'part1.png', canvas )

def hw3_2():
    img = cv2.imread( './input/QR_code.jpg' )
    w, h = 500, 500
    df = pd.read_csv('./QR_code_block.csv').columns.values[1:]
    QR_position = df.astype(np.int).reshape(4, 2)
    new_position = np.array( [[0, 0], [0, h], [h, 0], [h, h]] )
    # new_position = np.array( [[1480, 736], [2480, 736], [1480, 1736], [2480, 1736]] )
    # transform matrix
    warp = solve_homography( QR_position.astype( np.float32 ), new_position.astype( np.float32 ) )
    # warpPerspective
    img_warp = inverse_warpPerspective( img, warp, (h, h) )
    # save
    cv2.imwrite('part2.png', img_warp)

def hw3_3():
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    h, w, ch = img_front.shape
    df = pd.read_csv( './crosswalk_front_block.csv' ).columns.values[1:]
    crosswalk_position = np.array( [[340, 230], [387, 229], [340, 261], [387, 261]] )
    new_position = np.array( [[340, 230], [387, 229], [340, 261 + 40], [387, 261 + 40]] )  # original position:[[340 230], [387 229], [340 263], [387 261]]
    # transform matrix
    warp = solve_homography( crosswalk_position.astype( np.float32 ), new_position.astype( np.float32 ) )
    # warpPerspective
    # img_warp = cv2.warpPerspective( img_front, warp, (w, h) )
    img_warp = warpPerspective_p3( img_front, warp, (w, h) )
    # save
    cv2.imwrite( 'part3.png', img_warp )

    # # test
    # for i in range(0, 300, 20):
    #     print(i)
    #     new_position = np.array( [[340, 230], [387, 229], [340, 261+i], [387, 261+i]] )   # original position:[[340 230], [387 229], [340 263], [387 261]]
    #     # transform matrix
    #     warp = solve_homography( crosswalk_position.astype( np.float32 ), new_position.astype( np.float32 ) )
    #     # warpPerspective
    #     img_warp = cv2.warpPerspective( img_front, warp, (w, h) )
    #     cv2.imwrite( 'part3_test_'+str(i)+'_.png', img_warp )

def main():
    # Part 1
    ts = time.time()
    hw3_1()
    te = time.time()
    print( 'Elapse time: {}...'.format( te - ts ) )

    # Part 2
    ts = time.time()
    hw3_2()
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    ts = time.time()
    hw3_3()
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
    exit()

if __name__ == '__main__':
    main()
