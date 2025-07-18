import cv2, numpy as np, sys

def align_images(img1, img2):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
    return cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

def detect_and_save(p1, p2):
    img1 = cv2.imread(p1, 0)
    img2 = cv2.imread(p2, 0)
    aligned = align_images(img1, img2)
    diff = cv2.absdiff(img1, aligned)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    coords = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
        coords.append((x,y,x+w,y+h))
    cv2.imwrite('output.jpg', out)
    with open('changes.txt','w') as f:
        for x1,y1,x2,y2 in coords:
            f.write(f"{x1},{y1},{x2},{y2}\n")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("사용법: detect_changes.exe 이미지1 이미지2")
    else:
        detect_and_save(sys.argv[1], sys.argv[2])
