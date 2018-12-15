'''
  File name: morph_tri.py
  Author: John Wallison
  Date created:
'''

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
import matplotlib.pyplot as plt

# call vectorized version of the function
def morph_tri(source, target, source_points, target_points, warp_frac, dissolve_frac):
    return morph_tri_vectorized(source, target, source_points, target_points, warp_frac, dissolve_frac)

def morph_tri_vectorized(source, target, source_points, target_points, warp_frac, dissolve_frac):
    # source = source.transpose((1,0,2))
    # target = target.transpose((1,0,2))

    # Initialize stuff
    num_frames = warp_frac.shape[0]
    morphed_frames = np.zeros((num_frames, source.shape[0], source.shape[1], source.shape[2]))

    # OUTER LOOP: do this num_frames times
    for frame in np.arange(0, num_frames):
        warp = warp_frac[frame]
        diss = dissolve_frac[frame]
        # print("FRAME", (frame+1), "OF", num_frames)

        # Get intermediate points and triangulation
        '''print("Preparing triangles and stuff...")'''
        intermediate_points = (1 - warp) * source_points + (warp) * target_points
        triangle = Delaunay(intermediate_points)    # triangulation object

        # Generate vertices (pixels) of each triangle from simplices
        simplices = triangle.simplices
        num_simplices = simplices.shape[0]
        x,y = intermediate_points[simplices,0], intermediate_points[simplices,1]
        vertices = np.array([x, y]) #vertices[x/y, num_tris, a/b/c]

        # Do the same for source and target images
        x,y = source_points[simplices,0], source_points[simplices,1]
        source_vertices = np.array([x, y])
        x,y = target_points[simplices,0], target_points[simplices,1]
        target_vertices = np.array([x, y])

        # Compute inverses for each triangle (we will save them so we only have to do this once per simplex)
        '''print("Computing inverses...")'''
        inverses = np.zeros((num_simplices, 3, 3))  #inverses[num_tris, 3, 3]
        for i in np.arange(0, num_simplices):
            B = np.array([
                vertices[0][i],
                vertices[1][i],
                [1, 1, 1]
            ])
            inverses[i] = np.linalg.inv(B)

        ''' Generate stacks of matrices '''
        h, w = source.shape[0:2]
        n = h * w

        '''print("Generating stacks of matrices...")'''
        inverse_stack    = np.zeros((n, 3, 3))      # the stack of inverses of square B matrices
        source_mat_stack = np.zeros((n, 3, 3))      # B matrices corresponding to original source image
        target_mat_stack = np.zeros((n, 3, 3))      # like above, but for target image
        column_vec_stack = np.zeros((n, 3, 1))      # stack of [x, y, 1] vectors

        # Preparing indices and such we need to vectorize the loop: y,x order for mesh
        mesh_x, mesh_y = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')
        list_of_xs, list_of_ys = np.ravel(mesh_x).reshape((1,n)), np.ravel(mesh_y).reshape((1,n))
        stack_of_xys = np.concatenate((list_of_xs, list_of_ys), axis=0)

        ''' Prepare matrices for vectorized operations '''

        # Get the simplex stack (which triangle are we in for a given y,x pair)
        pair_xys = np.dstack((mesh_x, mesh_y))
        simplex_stack = np.apply_along_axis(triangle.find_simplex, 2, pair_xys)     # simplex_stack[x,y] = simple of pixel (x,y)
        simplices_ravel = np.ravel(simplex_stack)

        # Inverse stack
        inverse_stack = inverses[simplices_ravel]

        # Source matrix stack
        mat_src_row1 = source_vertices[0][simplices_ravel].reshape((n,3,1)).transpose((0,2,1))
        mat_src_row2 = source_vertices[1][simplices_ravel].reshape((n,3,1)).transpose((0,2,1))
        source_mat_stack = np.concatenate((mat_src_row1, mat_src_row2, np.ones(3*n).reshape((n,1,3))), axis=1)

        # Target matrix stack
        mat_tgt_row1 = target_vertices[0][simplices_ravel].reshape((n,3,1)).transpose((0,2,1))
        mat_tgt_row2 = target_vertices[1][simplices_ravel].reshape((n,3,1)).transpose((0,2,1))
        target_mat_stack = np.concatenate((mat_tgt_row1, mat_tgt_row2, np.ones(3*n).reshape((n,1,3))), axis=1)

        # Generate the column vectors
        column_vec_stack = np.concatenate(
            (stack_of_xys, np.ones(n).reshape((1,n))), axis=0
        ).reshape((3,n,1)).transpose((1,0,2))

        ''' Perform matrix multiplication to solve linear alg '''
        '''print("Doing matrix multiplication...")'''
        barycentric_coordinates_stack = np.matmul(inverse_stack, column_vec_stack)
        solution_source = np.matmul(source_mat_stack, barycentric_coordinates_stack)
        solution_target = np.matmul(target_mat_stack, barycentric_coordinates_stack)

        ''' Unravel solution stacks to generate the warped images '''
        '''print("Preparing this frame...")'''
        warped_source = np.zeros(source.shape)
        warped_target = np.zeros(source.shape)

        (xs_src, ys_src, zs_src) = (solution_source[:,0,0], solution_source[:,1,0], solution_source[:,2,0] + 10e-6)
        x_morph_src = np.clip(np.round(xs_src / zs_src), 0, h-1).astype(int)
        y_morph_src = np.clip(np.round(ys_src / zs_src), 0, w-1).astype(int)
        morphed_mesh_src = (x_morph_src.reshape((h,w)), y_morph_src.reshape((h,w)))
        # print("source", source.shape)
        # print("warped_source", warped_source.shape)
        # print("mesh_x", mesh_x.shape)
        # print("morphed_mesh_src", morphed_mesh_src[0].shape)
        warped_source[mesh_x, mesh_y] = source[morphed_mesh_src]

        (xs_tgt, ys_tgt, zs_tgt) = (solution_target[:,0,0], solution_target[:,1,0], solution_target[:,2,0] + 10e-6)
        x_morph_tgt = np.clip(np.round(xs_tgt / zs_tgt), 0, h-1).astype(int)
        y_morph_tgt = np.clip(np.round(ys_tgt / zs_tgt), 0, w-1).astype(int)
        morphed_mesh_tgt = (x_morph_tgt.reshape((h,w)), y_morph_tgt.reshape((h,w)))
        warped_target[mesh_x, mesh_y] = target[morphed_mesh_tgt]

        # warped_source = warped_source.transpose((1,0,2))
        # warped_target = warped_target.transpose((1,0,2))

        # Finally, dissolve to find the average image
        '''print("Dissolving...")'''
        intermediate_image = (1 - diss) * warped_source + (diss) * warped_target

        # Save this frame
        morphed_frames[frame,:,:,:] = intermediate_image#.astype(np.uint8)

    return morphed_frames

