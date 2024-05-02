import cv2 as cv

FILTER_SIZE = 15
IMAGE_VALUE_BACKGROUD = 3

class Processor:
    def __init__(self, enable_geometry, target_size, enable_graphy=False):
        self.__enable_geometry = enable_geometry
        self.__target_size = target_size
        self.__enable_graphy = enable_graphy
    
    def __find_largest_contour(self, mask_image):
        contours, _ = cv.findContours(mask_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)        
        y_min = 0
        x_min = 0
        y_max = mask_image.shape[0]
        x_max = mask_image.shape[1]
        
        if len(contours) > 0:
            max_area = 0
            for contour in contours:
                current_area = cv.contourArea(contour)
                if current_area >= max_area:
                    max_area = current_area
                    max_contour = contour
            
            x_min, y_min, w, h = cv.boundingRect(max_contour)
            x_max = x_min + w
            y_max = y_min + h
        
        return y_min, y_max, x_min, x_max
        
    def __remove_background(self, input_image, filter_size=FILTER_SIZE):
        image_grayscale = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
        image_median = cv.medianBlur(image_grayscale, filter_size)
        _, mask_image = cv.threshold(image_median, IMAGE_VALUE_BACKGROUD, 255, cv.THRESH_BINARY)
        y_min, y_max, x_min, x_max = self.__find_largest_contour(mask_image)
        
        result_image = input_image[y_min: y_max, x_min:x_max, :]
                                        
        return result_image

    def __resize_keep_ration(self, image, size, padding_color=(0, 0, 0)):    
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(size))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv.resize(image, new_size)
        delta_w = size[0] - new_size[0]
        delta_h = size[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=padding_color)
        
        return image
        
    def __process_geometry(self, input_image, size):    
        #Image crop to remove black background, then resize with keep-ratio    
        crop_image = self.__remove_background(input_image)        
        result_image = self.__resize_keep_ration(crop_image, (size, size))
            
        return result_image
    
    def __process_graphic(self, input_image):             
        result_image = input_image
            
        return result_image
    
    def process(self, input_image):
        result_image = input_image
        if self.__enable_geometry:
            result_image = self.__process_geometry(input_image, self.__target_size)
        if self.__enable_graphy:
            result_image = self.__process_graphic(input_image)
            
        return result_image