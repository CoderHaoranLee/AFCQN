from light_detection import *
army_color = "B"
class Armor_Detection:
    def __init__(self, img):
        self.img = img
        self.army_color = army_color
        self.debug_label = 0
        self.detect_None = False


    def get_light_area_1(self):
        b, g, r = cv2.split(self.img)
        if self.army_color == "R":
            thresh = 50
            result_img = cv2.subtract(r, g)
        else:
            result_img = cv2.subtract(b, g)
            thresh = 20
        ret, result_img = cv2.threshold(result_img, thresh, 255, cv2.THRESH_BINARY)
        index_range = np.argwhere(result_img == 255)
        if len(index_range) == 0:
            self.detect_None = True
            print "no light area"
            return [], []
        y_down = index_range[:, 0].max()
        y_up = index_range[:, 0].min()
        x_right = index_range[:, 1].max()
        x_left = index_range[:, 1].min()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result_img = cv2.dilate(result_img, kernel)
        self.mask_1 = [x_left, y_up, x_right, y_down]
        self.light_area = result_img
        self.w = x_left
        self.h = y_up
        # cv2.imshow("color",result_img)
        # cv2.waitKey()
        return result_img, [x_left, y_up, x_right, y_down]

    def get_light_area_2(self):
        if army_color == "B":
            lower_blue = np.array([0, 0, 200])
            upper_blue = np.array([190, 70, 255])
        else:
            #
            lower_blue = np.array([20, 0, 200])
            upper_blue = np.array([190, 70, 255])

        # cv2.imshow('Capture', img)  # change to hsv model
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)  # get mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue, )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask = cv2.erode(mask, kernel)
        self.mask_2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_light(self):
        result_2, box_point = self.get_light_area_1()
        result_1 = self.get_light_area_2()
        try:
            im = result_1[box_point[1]:box_point[3], box_point[0]:box_point[2]]
            self.light_img = self.img[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        except:
            self.detect_None = True
            return [], []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sure_bg = cv2.dilate(im, kernel, iterations=1)
        binary, contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            self.detect_None = True
            return [], []
        # if self.debug_label == 1:
        #     cv2.drawContours(im, contours, -1, (152, 152, 125), 5)
        light_list = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            light_list.append(box.tolist())
            # if self.debug_label == 1:
            #     cv2.drawContours(sure_bg, [box], 0, (152, 152, 125),3)
        armor_lights = []
        for light_i_points in light_list:
            light_i = Light(light_i_points)
            light_type = light_i.light_judge()
            if light_type == 1:
                armor_lights.append(light_i)
                # if self.debug_label == 1:
                #     cv2.drawContours(sure_bg, [np.array(light_i.point)], 0, (152, 152, 125), 3)
        if len(armor_lights) == 0:
            self.detect_None = 1
            print "no lights "
            return [], []
        return armor_lights, sure_bg

    def armor_filter(self):
        armor_lights, img = self.get_light()
        if self.detect_None == 1:
            return [], []
        armors = []
        for i in range(len(armor_lights) - 1):
            for j in range(i + 1, len(armor_lights)):
                armor_ij = Armor([armor_lights[i], armor_lights[j]])
                armor_ij.debug_label = self.debug_label
                armor_label = armor_ij.armor_judge()

                if armor_label == 1:
                     armors.append(armor_ij)

                else:
                    if self.debug_label == 1:
                        points = armor_lights[i].middle_line + armor_lights[j].middle_line[::-1]

        if len(armors) == 0:
            self.detect_None = 1
            print "no armor"
            return [], []
        return armors, img

def detection_main(im):
        armor_detect = Armor_Detection(im)
        result = armor_detect.armor_filter()
        if armor_detect.detect_None == 1:
            return []
        else:
            armors = result[0]

        # # if len(armors)>=1:
        armor_j = 0
        armors_2 = []
        for armor in armors:
            for i in range(len(armor.points)):
                armor.points[i] += np.array([armor_detect.w, armor_detect.h])
            # cv2.drawContours(im, [np.array(armor.points)], 0, (152, 152, 125), 3)
            im_ij = im[armor.line_area[1] +  armor_detect.h:armor.line_area[3] + armor_detect.h,
                    armor.line_area[0] +  armor_detect.w:armor.line_area[2] +  armor_detect.w]
            if 0 not in np.shape(im_ij):
                # points = [np.array([armor.line_area[0] +  armor_detect.w, armor.line_area[1] +  armor_detect.h]),
                #           np.array([armor.line_area[0] +  armor_detect.w, armor.line_area[3] +  armor_detect.h]),
                #           np.array([armor.line_area[2] +  armor_detect.w, armor.line_area[3] +  armor_detect.h]),
                #           np.array([armor.line_area[2] +  armor_detect.w, armor.line_area[1] +  armor_detect.h]),
                #           ]
                # t = cv2.resize(im_ij, (30, 30))

                # t = np.array([t])
                # a = sess.run(y_result, feed_dict={x: t, keep_prob: 1.})
                if 1:
                    armors_2.append(armor)
            else:
                armors_2.append(armor)
            armor_j += 1
        return im, armors_2


# if __name__ == "__main__":
#
#     weights = cnn_model.weights
#     biases = cnn_model.biases
#     x = tf.placeholder(tf.float32, [None, 30, 30, 3])
#     # y = tf.placeholder(tf.float32, [None, 2])
#     keep_prob = tf.placeholder(tf.float32)
#     pred = cnn_model.alex_net(x, weights, biases, keep_prob)
#     init = tf.initialize_all_variables()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, "/home/drl/catkin_ws/Model2/model.ckpt")
#         graph = tf.get_default_graph()
#         prediction = graph.get_tensor_by_name('add_2:0')
#         y_result = tf.nn.softmax(prediction)
#         name_list = os.listdir("error/")
#         for imname in name_list:
#         # for imname in range( 64,80):
#             # imname ='output/' + str(img_i) + '.jpg'
#             filename = 'error/' + imname
#             print imname, "-----------------"
#             im = cv2.imread(filename)
#             if im is None:
#                 print "read none"
#
#             else:
#                 # get_hsv(imname)
#                 # profile.run("detection_main(im)")
#
#                 result = detection_main(im)
#
#                 if len(result) != 0:
#                     im1 = result[0]
#                     armors = result[1]
#                     centre_points = []
#                     for armor in armors:
#
#                         distant = Distance()
#                         print "armor_points:", armor.points
#                         distant.get_distance(armor.points)
#                         centre_points.append(distant)
#                     # cv2.imshow("tt", im1)
#                     # cv2.imwrite("r3/" + str(img_i) + ".jpg", im1)
#                     cv2.imwrite("r3/" + imname, im1)
#                     print 2
#
#                     final_distance = filter_robo(centre_points, 30)