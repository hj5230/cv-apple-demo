import cv2

print("OpenCL Available:", cv2.ocl.haveOpenCL())

cv2.ocl.setUseOpenCL(True)

print("OpenCL Available:", cv2.ocl.useOpenCL())
