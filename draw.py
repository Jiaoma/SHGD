import numpy as np
import cv2

    
def drawBoxes(img,box,pid):
    if not isinstance(img,np.ndarray):
        img=img.detach().cpu().numpy()    
    frame_tmp = img
    left,top,right,bottom=[int(i) for i in box]
    cv2.rectangle(frame_tmp, (left, top), (right, bottom), (255, 0, 0), 2)
    
    label = '%d' % int(pid)
        
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
    top = max(top, labelSize[1])
    # frame_tmp = cv2.rectangle(frame_tmp, (left, int(top - round(1.5*labelSize[1]))), (left + int(round(1.5*labelSize[0])), top + baseLine), (255, 255, 255), cv2.FILLED)
    # frame_tmp = cv2.putText(frame_tmp, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 7.5, (0,0,0), 10)
    return frame_tmp