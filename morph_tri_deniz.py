'''
  File name: morph_tri.py
  Author: Deniz Beser
  Date created: 10.12.2018
'''
from scipy.spatial import Delaunay
'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: source image
    - Input im2: target image
    - Input im1_pts: correspondences coordiantes in the source image
    - Input im2_pts: correspondences coordiantes in the target image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''
import numpy as np
from scipy.spatial import Delaunay

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  
  morphed_ims = []

  for itr in range(len(warp_frac)):
    t = 1-warp_frac[itr]
    d = 1-dissolve_frac[itr]
    # Get Delaunay for images and average image
    ave_pts = [((t*im1_pts[i][0]+(1-t)*im2_pts[i][0]),(t*im1_pts[i][1]+(1-t)*im2_pts[i][1])) for i in range(len(im1_pts))]
    triAve = Delaunay(ave_pts)

    # Pre compute Matrices
    tri1toA = {}
    tri2toA = {}
    triAvetoAinv = {} 
    for triNum in range(triAve.simplices.shape[0]):
      cor1 = triAve.simplices[triNum]
      cor2 = triAve.simplices[triNum]
      # get corner coordinates
      sa,sb,sc = im1_pts[cor1[0]], im1_pts[cor1[1]], im1_pts[cor1[2]]
      A1 = np.array([[sa[0],sb[0],sc[0]],[sa[1],sb[1],sc[1]],[1, 1, 1]])
      sa,sb,sc = im2_pts[cor2[0]], im2_pts[cor2[1]], im2_pts[cor2[2]]
      A2 = np.array([[sa[0],sb[0],sc[0]],[sa[1],sb[1],sc[1]],[1, 1, 1]])
      tri1toA[triNum] = A1
      tri2toA[triNum] = A2

    for triNum in range(triAve.simplices.shape[0]):
      # get corners
      corners = triAve.simplices[triNum]
      # get corner coordinates
      a,b,c = ave_pts[corners[0]], ave_pts[corners[1]], ave_pts[corners[2]]
      # Solve for inverse intermediate matrix
      Ainv = np.linalg.inv(np.array([[a[0],b[0],c[0]],[a[1],b[1],c[1]],[1, 1, 1]]))
      triAvetoAinv[triNum] = Ainv

    # height and width of target  
    h,w = int(t*im1.shape[0]+(1-t)*im2.shape[0]),int(t*im1.shape[1]+(1-t)*im2.shape[1])
    new1 = np.zeros((h,w,3), dtype=int)
    new2 = np.zeros((h,w,3), dtype=int)

    # Go over each pixel
    for i in range(w):
      for j in range(h):
        # Get which triangle
        triNum = int(triAve.find_simplex(np.array([i,j])))
        if triNum < 0: continue
        b = np.array([i,j,1])
        bar = np.matmul(triAvetoAinv[triNum],b)

        # Get coordinates of source matrices
        pos = np.matmul(tri1toA[triNum],bar)
        x, y, z = pos[0],pos[1],pos[2]
        x,y = int(x/z),int(y/z)
        new1[j,i,:] = im1[y,x,:]

        pos = np.matmul(tri2toA[triNum],bar)
        x, y, z = pos[0],pos[1],pos[2]
        x,y = int(x/z),int(y/z)
        new2[j,i,:] = im2[y,x,:]

    morphed_im = (np.multiply(new1,d)  + np.multiply(new2,1-d)).astype(int)
    # print(morphed_im.shape)
    morphed_ims.append(morphed_im)
    
  return np.array(morphed_ims).astype(np.uint8)
