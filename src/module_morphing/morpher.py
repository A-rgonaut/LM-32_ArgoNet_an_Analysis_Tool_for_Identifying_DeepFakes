
import cv2
import dlib
import numpy as np

class Morpher():
    """
    Morpher is a class designed for facial morphing between two images. It utilizes facial landmarks 
    to align and morph faces, allowing for smooth transitions between the source and destination images.

    Available subpackages
    -
     - cv2 
     - dlib 
     - numpy 

    Attributes
    -
    t   (float)
        The morphing factor, determining the blend between the source and destination images.
    src (cv2.typing.MatLike)
        The source image.
    dst (cv2.typing.MatLike)
        The destination image.
    row (int)
        The number of rows (height) in the source image.
    col (int)
        The number of columns (width) in the source image.
    detector (dlib.fhog_object_detector)
        The face detector used to identify faces in the images.
    predictor (dlib.shape_predictor)
        The landmark predictor used to extract facial landmarks.

    Methods
    -
     - swap_src_dst()
     - morph(warp_t: float = None, only_face: bool = False, video: bool = False) -> np.ndarray
     - __crop_faces(self)
     - __get_landmarks(img: cv2.typing.MatLike, only_face: bool) -> list
     - __get_triangulation(points: np.ndarray) -> np.ndarray
     - __get_indexes(points: np.ndarray, triangles: list) -> list
     - __warp(src_triangles_indexes: list, src_points: np.ndarray, int_points: np.ndarray, flag: bool = True) -> np.ndarray
     - __create_video(filename: str, frames: list) 
    """

    def __init__(self, src : cv2.typing.MatLike, dst : cv2.typing.MatLike, t : float = 0.5):
        self.t = t
        self.src = src
        self.dst = dst
        self.row, self.col, _ = src.shape
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./src/shape_predictor_68_face_landmarks.dat")
    
    def swap_src_dst(self):
        """
        Exchanges the values of the `src` and `dst` attributes, swapping their roles.
        """
        self.src, self.dst = self.dst, self.src
    
    def __crop_faces(self):
        """
        Crops the faces detected in the source and destination images and resizes the destination image
        to match the dimensions of the source image.
        This method uses a face detector to identify the bounding boxes of faces in the source (`self.src`)
        and destination (`self.dst`) images. It then applies a 'margin' to the bounding boxes to include
        additional surrounding areas, crops the images accordingly, and resizes the destination image
        to match the dimensions of the cropped source image.

        Attributes Modified:
        -
         - `self.src`: Cropped version of the source image.
         - `self.dst`: Cropped and resized version of the destination image.
         - `self.row`: Number of rows (height) in the cropped source image.
         - `self.col`: Number of columns (width) in the cropped source image.

        Note:
        -
         - The method assumes that the face detector returns a list of bounding boxes, and the first
           bounding box corresponds to the primary face in the image.
         - The margin applied to the bounding box is defined by the `margin` variable.
         - The cropped images are converted to `np.uint8` type before resizing.
        """
        # Detecting the rectangles of faces in the images 
        src_face = self.detector(self.src)[0]
        dst_face = self.detector(self.dst)[0]

        # Setting a margin to include additional surrounding areas
        margin = 1.65

        coords = []

        # Calculating the new coordinates of rectangles based on margin
        for img in (src_face, dst_face):

            x, y, w, h = img.left(), img.top(), img.right(), img.bottom()
        
            x_size = int(w * margin)
            y_size = int(h * margin)
        
            x_cent = x + w // 2
            y_cent = y + h // 2

            x_new = max(0, x_cent - x_size // 2)
            y_new = max(0, y_cent - y_size // 2)

            coords.append((y_new, y_size, x_new, x_size))

        # Cropping the faces based on the new coordinates
        self.src = self.src[coords[0][0]:coords[0][1], coords[0][2]:coords[0][3]].astype(np.uint8)
        self.dst = self.dst[coords[1][0]:coords[1][1], coords[1][2]:coords[1][3]].astype(np.uint8)

        # Recalculating the row and col parameters
        self.row, self.col, _ = self.src.shape

        # Resizing the destination image to match the source image dimensions
        self.dst = cv2.resize(self.dst, (self.col, self.row))
    
    def __get_landmarks(self, img : cv2.typing.MatLike):
        """
        Extracts facial landmarks and image corner points from the given image.
        This method uses a face detector to identify faces in the image and a landmark predictor 
        to extract 68 facial landmarks for each detected face. Additionally, it appends the 
        corner points of the image to the landmarks.
        Args:
            img (cv2.typing.MatLike): The input image from which landmarks are to be extracted.
        Returns:
            list: A list of numpy arrays, where each array contains the facial landmarks 
                  (as (x, y) tuples) and the corner points of the image for each detected face.
        """
        faces = []

        # dlib.detector() detects multiple faces in the image and returns a rectangle for each face
        for face in self.detector(img):
            # dlib.predictor() predicts the landmarks of the face in the image and returns a list of landmarks
            landmarks = self.predictor(img, face)

            # Converting the landmarks in a list of tuple (x,y) 
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            
            # Extending the landmarks list with the corners points of the image 
            points.extend([(0, 0), (self.col - 1, 0), (0, self.row - 1), (self.col - 1, self.row - 1)])

            # Appending the landmarks list to the faces list
            faces.append(np.array(points, dtype=np.int32))

        return faces

    def __get_triangulation(self, points : np.ndarray):
        """
        Computes the Delaunay triangulation for a given set of points.
        This method creates a rectangular bounding box based on the dimensions
        of the image (self.col and self.row) and initializes a cv2.Subdiv2D object
        for Delaunay triangulation. The provided points are inserted into the
        subdivision, and the resulting triangulation is returned as a list of triangles.
        Args:
            points (numpy.ndarray): A NumPy array of shape (N, 2) containing the 
                                    coordinates of the points to be triangulated.
        Returns:
            numpy.ndarray: A NumPy array of shape (M, 6) where each row represents
                           a triangle defined by the coordinates of its three vertices
                           in the format [x1, y1, x2, y2, x3, y3].
        """
        rect = (0, 0, self.col, self.row)
        # Create an empty subdiv
        subdiv = cv2.Subdiv2D(rect)
        # Insert points into subdiv
        subdiv.insert(points.astype(np.float32))

        return subdiv.getTriangleList().astype(np.int32)

    def __get_indexes(self, points : np.ndarray, triangles : list):
        """
        Extracts the indexes of points corresponding to the vertices of triangles.
        This method takes a list of points and a list of triangles (defined by their vertex coordinates)
        and returns a list of triangles represented by the indexes of their vertices in the points array.
        Args:
            points (numpy.ndarray): A 2D array of shape (N, 2) where each row represents a point (x, y).
            triangles (list of lists): A list of triangles, where each triangle is represented by 
                a list of six values [x1, y1, x2, y2, x3, y3] corresponding to the coordinates of its vertices.
        Returns:
            (list of lists): A list of triangles, where each triangle is represented by a list of three indexes 
            [`index_pt1`, `index_pt2`, `index_pt3`] corresponding to the positions of its vertices in the points array.
            Only triangles with all vertices found in the points array are included in the result.
        """
        indexes = []

        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # Extract indexes of the points
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = index_pt1[0][0] if len(index_pt1[0]) > 0 else None

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = index_pt2[0][0] if len(index_pt2[0]) > 0 else None

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = index_pt3[0][0] if len(index_pt3[0]) > 0 else None
            
            # Adding the triangle if all points are found
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                indexes.append([index_pt1, index_pt2, index_pt3])

        return indexes

    def __warp(self, src_triangles_indexes, src_points, int_points, flag : bool = True):
        """
        Performs affine warping of triangles from a source image to a intermediate image.
        Args:
            src_triangles_indexes (list of list of int): List of triangle vertex indices 
                corresponding to the source points.
            src_points (numpy.ndarray): Array of points representing the vertices of the 
                source image triangles.
            int_points (numpy.ndarray): Array of points representing the vertices of the 
                intermediate or destination image triangles.
            flag (bool, optional): Determines the direction of warping. If True, warps 
                from `self.src` to `self.dst`. If False, warps from `self.dst` to `self.src`. 
                Defaults to True.
        Returns:
            numpy.ndarray: The output image with the warped triangles applied.
        """
        output_img = np.zeros_like(self.src, dtype=np.uint8)

        for triangle_index in src_triangles_indexes:
            # Triangulation of the first face
            tr1_pt1 = src_points[triangle_index[0]]
            tr1_pt2 = src_points[triangle_index[1]]
            tr1_pt3 = src_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            (x, y, w, h) = cv2.boundingRect(triangle1)
            cropped_triangle = (self.src if flag else self.dst)[y: y + h, x: x + w]
            cropped_tr1_mask = cv2.cvtColor(cropped_triangle, cv2.COLOR_BGR2GRAY)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                               [tr1_pt2[0] - x, tr1_pt2[1] - y],
                               [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
            cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

            # Triangulation of second face
            tr2_pt1 = int_points[triangle_index[0]]
            tr2_pt2 = int_points[triangle_index[1]]
            tr2_pt3 = int_points[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            (x, y, w, h) = cv2.boundingRect(triangle2)
            cropped_triangle2 = (self.dst if flag else self.src)[y: y + h, x: x + w]
            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
            cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)

            M = cv2.getAffineTransform(points, points2)
            
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            # Getting the area of the triangle in the output image
            triangle_area = output_img[y: y + h, x: x + w]

            # Applying the mask to remove white borders
            triangle_area_bg = cv2.bitwise_and(triangle_area, triangle_area, mask=cv2.bitwise_not(cropped_tr2_mask))
            warped_triangle_fg = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
            
            # Combining the transformed triangle with the output area
            output_img[y: y + h, x: x + w] = cv2.add(triangle_area_bg, warped_triangle_fg)

        return output_img

    def morph(self, warp_t : float = None, video : bool = False):
        """
        Morphs between two images based on facial landmarks and triangulation.
        This method performs image morphing by cropping, aligning the faces of two images 
        by calculating intermediate points, and blending the images based on a warp scale. 
        Optionally, it can generate a video showing the morphing process.
        Args:
            warp_t (float, optional): The warp scale that determines the face morphology 
                alignment of the images. A value of 0 aligns the images to the source ('src') 
                image, while a value of 1 aligns them to the destination ('dst') image. 
                Defaults to the instance variable `self.t` if not provided.
            video (bool, optional): If True, generates a video showing the morphing process. 
                Defaults to False.
        Returns:
            numpy.ndarray: The morphed image.
        Notes:
        -
         - The method assumes that there is only one face in each image.
         - If `video` is True, the resulting video is saved to 'res/result.mp4'.
        """
        # Setting the warp scale that determines the face morphology alignment of the images
        # similar to the 'src' image (0) or the 'dst' image (1). By default, the value is 't' 
        warp_t = self.t if warp_t is None else warp_t

        # Cropping the faces
        self.__crop_faces()

        # Getting the image landmarks.
        # The method returns multiple faces landmarks, we suppose to have only one face in the image 
        src_points = self.__get_landmarks(self.src)[0]
        dst_points = self.__get_landmarks(self.dst)[0]

        # Getting the triangulation of the faces 
        src_triangles = self.__get_triangulation(src_points)
        dst_triangles = self.__get_triangulation(dst_points)

        # Getting the indexes of the triangles 
        src_triangles_indexes = self.__get_indexes(src_points, src_triangles)
        dst_triangles_indexes = self.__get_indexes(dst_points, dst_triangles)

        # Getting the intermediate points to align the faces
        int_points = cv2.addWeighted(src_points, 1 - warp_t, dst_points, warp_t, 0)

        # Warping the images 
        src_morph = self.__warp(src_triangles_indexes, src_points, int_points, True)
        dst_morph = self.__warp(dst_triangles_indexes, dst_points, int_points, False)

        # Applying the morphing 
        img_morph = cv2.addWeighted(src_morph, 1 - self.t, dst_morph, self.t, 0)

        # Generating the video if required 
        if video:
            # Generation the Triangulation-Warping frames of the source
            warp_src_frames = [
                self.__warp(src_triangles_indexes, src_points, cv2.addWeighted(src_points, 1 - t, dst_points, t, 0), True)
                for t in np.arange(0, warp_t, 1/60)]
            #self.__create_video('res/warp_src_int.mp4', warp_src_frames)

            # Generation the Triangulation-Warping frames of the destination ""
            warp_dst_frames = [
                self.__warp(dst_triangles_indexes, dst_points, cv2.addWeighted(dst_points, 1 - t, src_points, t, 0), False)
                for t in np.arange(1 - warp_t, 0, -1/60)]
            #self.__create_video('res/warp_dst_int.mp4', warp_dst_frames)

            # Generation of Blending Frames 
            morph_frames = [cv2.addWeighted(src_morph, 1 - t, img_morph, t, 0) for t in np.arange(0, 1, 1/60)] \
                         + [cv2.addWeighted(img_morph, 1 - t, dst_morph, t, 0) for t in np.arange(0, 1, 1/60)]
            #self.__create_video('res/morphing.mp4', morph_frames)

            # Rendering the video 
            self.__create_video('./res/result.mp4', warp_src_frames + morph_frames + warp_dst_frames)

        return img_morph

    def __create_video(self, filename: str, frames: list):
        """
        Creates a video file from a sequence of frames on the current working directory.
        Args:
            filename (str): The name of the output video file, including its path.
            frames (list): A list of frames (images) to be written into the video.
        """
        video = cv2.VideoWriter(filename, -1, 60, (self.col, self.row))
        for frame in frames:
            video.write(frame)
        video.release()