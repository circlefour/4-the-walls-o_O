import cv2 as cv
import numpy as np
import qrcode
import concurrent.futures

def generate_qr_code(data='insert the coolest website ever here'):
    qr = qrcode.QRCode(border=3, box_size=3)
    qr.add_data(data)
    qr.make()
    return np.array(qr.make_image(fill_color='red', back_color="#161FE7"))

def apply_face_glitch(input, faces, glitch_functions):
    if faces[1] is not None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            glitched_faces = []
            for face in faces[1]:
                coords = face[:-1].astype(np.int32)
                x0, y0, x1, y1 = coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]
                
                # shallow copy
                just_face = input[y0:y1, x0:x1]
                
                # checks if the width or height of the face is 0 (if a face wasn't recognized, or region is too small such as at edges of screen)
                if just_face.shape[0] == 0 or just_face.shape[1] == 0:
                    continue

                # applying multiple glitches
                for glitch_function, glitch_args in glitch_functions:
                    glitched_face = executor.submit(glitch_function, just_face, **glitch_args)
                glitched_faces.append((glitched_face, (y0, y1), (x0, x1)))

            # place glitched faces over original faces
            for glitched_face, y_range, x_range in glitched_faces:
                augmented_face = glitched_face.result()
                input[y_range[0]:y_range[1], x_range[0]:x_range[1]] = augmented_face

def apply_frame_filter(frame, frame_filters):
    for filter_function, filter_args in frame_filters:
        frame = filter_function(frame, **filter_args)
    return frame