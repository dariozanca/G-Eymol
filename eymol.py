'''
Created on:	20 Dec 2019

@author: 	Dario Zanca, PhD (dario.zanca@unisi.it, dariozanca@gmail.com)

            Post-doc @ University of Siena, Dept. of Medicine, Surgery and Neuroscience.


@summary: 	Collection of functions to generate scanpaths with G-EYMOL.
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import numpy as np
import cv2
from math import sin, pi, isnan
from random import randint, uniform
from scipy.integrate import odeint
import time
# IMPORT EXTERNAL LIBRARIES
import os
import csv

########################################################################################################################
########################################################################################################################

''' 
Main class to create an istance of the model.

Example of use:

    params = {'alpha_c': 0.1, 'alpha_of': 0.2, 'max_distance': 300}
      
    foa = Eymol(params)
    
    for t in range(T):
        foa.next_location(frame_t, of_t)        
'''

class Eymol():

    def __init__(self, parameters):

        ''' parameters: it is a dictionary of parameters.
                'alphas':           a list of weights, one for each channel.

                'max_distance':     maximum distance from actual point to consider in the integral
                                    suggested value average image dimensions
                'dissipation':      weigth of the term of dissipation
                                    suggested value 0.1
                'frame_rate':       frame per second of the input video stream

                'h_w':              frame size list

                'is_online':        True if you argoing with webcam, False otherwise
        '''

        # Initial state
        self.t = 0
        self.y = []

        # Parameters
        self.parameters = parameters

        max_d = parameters['max_distance']

        ### self.is_online = parameters['is_online']
        self.is_online = False 

        self.frame_rate = parameters['fps']
        self.h, self.w = parameters['h'], parameters['w']

        self.saccades_per_second = 3.
        self.real_time_last_saccade = time.clock()

        # Generate distances matrix
        self.distances_matrix = create_distances_matrix( max_d )

        # Generate a matrix to mark pixel to which inhibit return
        self.IOR_matrix = np.zeros( (self.h, self.w) )


    def next_location(self, feature_maps):

        '''
            Input:
                frame_t: RGB image
                of_t: optical flow (2 channels)

            Output:
                y = [row, column, row velocity, column velocity] of the next location
        '''

        self.y = compute_next_location(
                            # Visual input
                            feature_maps = feature_maps,

                            # Initial condition of the system and time instants to integrate
                            y0 = self.y,
                            times = np.arange(self.t, self.t + 1, .1),

                            # System parameters
                            parameters = self.parameters,

                            distances_matrix = self.distances_matrix,

                            IOR_matrix = self.IOR_matrix
                            )

        self.t += 1


        # TODO: pezza momentanea
        # restituisci solo pixel dentro il frame
        y_out = self.y
        y_out[0], y_out[1] = stayinside(feature_maps[0], row_col=y_out[0:2])

        # add pixel to the inhibition of return matrix
        if not self.is_online:
            if self.t % int(self.frame_rate / self.saccades_per_second) == 0:
                self.IOR_matrix = inhibit_return_in(self.IOR_matrix, row_col=y_out[0:2])

        else:
            if time.clock() - self.real_time_last_saccade >= (1. / self.saccades_per_second):
                self.IOR_matrix = inhibit_return_in(self.IOR_matrix, row_col=y_out[0:2])
                self.real_time_last_saccade = time.clock() # update real time of the last saccade

        return y_out

    def reset(self, y=[]):

        # Initial state
        self.t = 0
        self.y = y

########################################################################################################################
########################################################################################################################

def compute_next_location(
                            # Visual input
                            feature_maps,

                            # Initial condition of the system and 
                            # time instants to integrate
                            times,
                            y0,

                            # System parameters
                            parameters,

                            distances_matrix,

                            IOR_matrix
                            ):

    ''' Given input feature maps, this function returns the next location of the visual
        attention scanpath '''

    "Get feature maps dimensions"
    h, w = feature_maps[0].shape

    # "Add parameters"
    # parameters['k'] = 10**6
    # r = 0
    # parameters['r'] = (r, h - r, r, w - r)


    "Numerical method"

    # If not provided, generate random initial conditions
    if not y0:
        y0 = generate_initial_conditions(h,w)

    # Generate scanpath (by integrating diff. equations)
    y = odeint(myode, y0, times,
               args=(feature_maps, parameters, distances_matrix, IOR_matrix),
               mxstep=100, rtol=.1, atol=.1
               )

    return list(y[-1])

########################################################################################################################

def generate_initial_conditions(h,w):

    ''' This function generates initial condition for the dynamical system to be
    integrated. Numbers used here are arbitrary. Consider to motify or determine better
    numbers in future implementations. '''

    initRay = int(min(h, w) * 0.17)
    x1_init = int(h / 2) + randint(-initRay, initRay)
    x2_init = int(w / 2) + randint(-initRay, initRay)
    v1_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))
    v2_init = 2.0 * uniform(0.3, 0.7) * ((-1) ** randint(0, 1))

    return [x1_init, x2_init, v1_init, v2_init]

########################################################################################################################

def crop(frame, x_y, n):

    x, y = x_y

    if n % 2:
        d = (n//2)
    else:
        d = (n // 2) + 1

    h, w = np.shape(frame)

    if x < 0: x = 0
    elif x >= h: x = h-1

    if y < 0: y = 0
    elif y >= w: y = w-1

    x = int(x) + d
    y = int(y) + d

    frame = cv2.copyMakeBorder(frame,d,d,d,d,cv2.BORDER_CONSTANT,value=0)

    return frame[x-d:x+d+1, y-d:y+d+1]

########################################################################################################################

def myode(y, t, feature_maps, parameters, distances_matrix, IOR_matrix,

          apply_ior = True):

    '''	This function describes the system of two second-order differential
        equations which describe visual attention. (VERSION 3 - GRAVITATIONAL)

        y: it is the vector of the variables (x1, x2, dot x1, dot x2)

        t: time (frames)

        parameters: dictionary containing all the parameters of the model '''

    # Get parameters

    dissipation = parameters['dissipation']
    alphas = parameters['alphas']

    # create gradients channels

    channels = []

    for i in range(len(feature_maps)):

        gradient = get_gradients(feature_maps[i])

        channel = np.sqrt(gradient[:, :, 0]**2 + gradient[:, :, 1]**2)

        # Apply IOR function (Inhibition of Return)
        if apply_ior:
            channel *= (1 - IOR_matrix)

        channels.append(channel)

    # Apply distances matrix

    n = np.shape(distances_matrix)[1]

    channel_crops = []

    for channel in channels:

        channel_crop = crop(channel, (y[0], y[1]), n)

        if not channel_crop.max() == 0:
            channel_crop /= channel_crop.max()

        channel_crops.append(channel_crop)

    # define gravitational fields contributions

    C_x = []

    for i in range(len(channel_crops)):

        C_x.append(
            alphas[i] *   np.array(
                   [   (distances_matrix[0, :, :] * channel_crops[i]).sum(),

                       (distances_matrix[1, :, :] * channel_crops[i]).sum()    ]   )
        )


    "System of differential equations"

    dy = [  y[2],

            y[3],

            sum([C_x[i][0] for i in range(len(channel_crops))]) - dissipation * y[2],

            sum([C_x[i][1] for i in range(len(channel_crops))]) - dissipation * y[3],
          ]

    return dy

########################################################################################################################

def create_distances_matrix(n):

    ''' Create distances_mask for sum on the frame
        (x - a) / |x-a|**2
        notice: (x-a) is a vector.
        The resulting matrix is of dimension 2 x w x h. '''

    distances_matrix = np.zeros((2, n, n))

    center_x, center_y = (n//2), (n//2)

    for i in range(n):
        for j in range(n):
            if not (i == center_x and j == center_y):
                distances_matrix[0, i, j] = (n//10 + 1) * float(i - center_x) / (
                        ((i-center_x)**2 + (j - center_y)**2) + (n//10))

    for i in range(n):
        for j in range(n):
            if not (i == center_x and j == center_y):
                distances_matrix[1, i, j] = (n//10 + 1) * float(j - center_y) / (
                        ((i-center_x)**2 + (j - center_y)**2)  + (n//10))

    return distances_matrix


########################################################################################################################

def write_red_dot(frame, row_col,
                  RAY=5,
                  fixation_flag=False,
                  col_fix=(255, 0, 0),
                  col_sac=(0, 0, 255)):

    row, col = row_col

    # get point coordinates
    if isnan(row) or isnan(col):
        row, col = 0, 0
    else:
        row, col = int(row), int(col)

    if (row - RAY < 0):
        row = RAY
    else:
        if (row + RAY >= np.shape(frame)[0]):
            row = np.shape(frame)[0] - RAY - 1
    if (col - RAY < 0):
        col = RAY
    else:
        if (col + RAY >= np.shape(frame)[1]):
            col = np.shape(frame)[1] - RAY - 1

    if fixation_flag:
        cv2.circle(frame,
                   (col, row),
                   RAY, col_fix, 1)
    else:
        cv2.circle(frame,
                   (col, row),
                   RAY, col_sac, -1)

    return frame

########################################################################################################################

def gaussian(frame, row_col, RAY=25, blur=51):

    ''' This function returns a new frame with the same dimensions of frame, with a gaussian centered in the
        position (row, col).
        For a fast implementation, the gaussian is draw as a circle and then gaussian blurring is applied.  '''

    row, col = row_col[0], row_col[1]
    new_frame = np.zeros(np.shape(frame))
    cv2.circle(new_frame,
               (col, row),
               RAY, (1,), -1)
    new_frame = cv2.GaussianBlur(new_frame,(blur,blur),0)
    if not new_frame.max() == 0: new_frame /= new_frame.max()
    return new_frame

def inhibit_return_in(frame, row_col, RAY=35):

    row, col = stayinside(frame, row_col, RAY=RAY)

    new_frame = gaussian(frame, (row, col), RAY=RAY)

    frame = 0.9 * frame

    # add new inhibition signal
    frame += new_frame

    # Cut values greater than 1
    frame[frame>1] = 1.

    return frame


########################################################################################################################

def stayinside(frame, row_col, RAY=5):

    row, col = row_col

    # get point coordinates
    if isnan(row) or isnan(col):
        row, col = 0, 0
    else:
        row, col = int(row), int(col)

    if (row - RAY < 0):
        row = RAY
    else:
        if (row + RAY >= np.shape(frame)[0]):
            row = np.shape(frame)[0] - RAY - 1
    if (col - RAY < 0):
        col = RAY
    else:
        if (col + RAY >= np.shape(frame)[1]):
            col = np.shape(frame)[1] - RAY - 1

    return row, col

########################################################################################################################

def get_gradients(frame_t):

    sobelx = cv2.Sobel(frame_t, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame_t, cv2.CV_64F, 0, 1, ksize=5)

    return np.dstack( (sobelx, sobely) )

########################################################################################################################

def euclidean_distance(x,y):

    sum = 0

    for i in range(len(x)):

        sum += (x[i] - y[i])**2

    return sum**.5

def wave(frame, t, T=25):

    ''' n: dimension of the squared frame
        T: period of the wave (in frames) '''

    # this is to have a complete period in "frame_rate" number of frames
    omega = (2 * pi) / T

    # get dimensions
    h, w = np.shape(frame)[0], np.shape(frame)[1]

    # get some parameters that depend on the image
    C = h//2, w//2 # center of the image
    L = euclidean_distance(C, (0,0)) # maximum distance from the center of the image

    # compute the wave function
    wave = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            wave[i, j] = sin(omega*t + (pi/2)*(euclidean_distance(C, (i,j))/L))**2

    return wave

def create_wave_matrix(h_w, T):

    ''' (h, w): dimensions of the frame
        T: period of the wave (in frames) '''

    h, w = h_w

    T = int(T)  # fix, hack?
    wave_batch = np.zeros((T,h,w))

    for t in range(T):

        wave_batch[t] = wave(wave_batch[t], t, T)

    return wave_batch

########################################################################################################################


def extract_basic_features(inputImage):

    # convert scale of array elements
    src = np.float32(inputImage) * 1./255

    # split
    (B, G, R) = cv2.split(src)

    # extract an intensity image
    I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # define orientation angles
    thetas = np.pi * np.array([.0, .25, .5, .75])

    # create gabor filters and create orientation maps
    orientation_channels = []
    for theta in thetas:
        g_kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        orientation_channels.append(cv2.filter2D(I, cv2.CV_8UC3, g_kernel))

    return [B, G, R, I] + orientation_channels



def compute_simulated_scanpath(STIMULUS, seconds=5, fps=25, alphas_coeff=.2):

    # Resize input stimulus to have maximum dimensione equal to "maximum_dimension"

    maximum_dimension = 224.

    h, w, _ = np.shape(STIMULUS)

    if h > w:
        w_new = int((maximum_dimension/h)*w)
        h_new = int(maximum_dimension)
    else:
        h_new = int((maximum_dimension/w)*h)
        w_new = int(maximum_dimension)


    # Generate feature maps for that stimulus
    STIMULUS = cv2.resize(STIMULUS, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    feature_maps = extract_basic_features(STIMULUS)

    # Create an Eymol object

    parameters = {}

    parameters['fps'] = fps
    parameters['h'], parameters['w'], _ = np.shape(STIMULUS)

    parameters['alphas'] = [alphas_coeff / len(feature_maps), ] * len(feature_maps)

    parameters['dissipation'] = 1.5

    if max(parameters['h'], parameters['w']) % 2:
        parameters['max_distance'] = max(parameters['h'], parameters['w'])
    else:
        parameters['max_distance'] = max(parameters['h'], parameters['w']) + 1

    foa = Eymol(parameters=parameters)

    # Simulate eye-movements for each frame

    number_of_frames = seconds*fps

    scanpath = np.zeros((number_of_frames, 3))


    for t in range(number_of_frames):

        y0_last = foa.next_location(feature_maps)

        scanpath[t, 0], scanpath[t, 1], scanpath[t, 2] = y0_last[0], y0_last[1], float(t)/fps

    # Upscale coordinates of the gaze position
    scanpath[:, 0] *= float(h) / h_new
    scanpath[:, 1] *= float(w) / w_new

    return scanpath

def save_scanpath(  SIMULATED_SCANPATH_FOLDER,
                    dataset_name, stimulus_filename,
                    scanpath_id,
                    scanpath):

    stimulus_name, _ = os.path.splitext(stimulus_filename)

    # saving path
    path = SIMULATED_SCANPATH_FOLDER
    if not os.path.exists(path):
        os.mkdir(path)

    path += '/' + dataset_name
    if not os.path.exists(path):
        os.mkdir(path)

    path += '/' + stimulus_name + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    # save the array as file
    np.savetxt(path + str(scanpath_id), scanpath)

    return


def write_on(filename, row):
    f = open(filename, 'a')
    f.write(row)
    f.close()

def read_scanpath_from_file(stimulus_folder, scanpath_id):

    with open(os.path.join(stimulus_folder, scanpath_id), 'r') as f:

        f_reader = csv.reader(f, delimiter=' ')

        eye_positions = []

        for line in f_reader:
            eye_positions.append(np.array(line).astype(float))

    return np.array(eye_positions)


def fixation_detection(x, y, time, maxdist=25, mindur=50):
    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data)

    arguments

    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of EyeTribe timestamps

    keyword arguments

    missing	-	value to be used for missing data (default = 0.0)
    maxdist	-	maximal inter sample distance in pixels (default = 25)
    mindur	-	minimal duration of a fixation in milliseconds; detected
                fixation cadidates will be disregarded if they are below
                this duration (default = 100)

    returns
    Sfix, Efix
                Sfix	-	list of lists, each containing [starttime]
                Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
    """

    # empty list to contain data
    Sfix = []
    Efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        squared_distance = ((x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2)
        dist = 0.0
        if squared_distance > 0:
            dist = squared_distance ** 0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], time[i - 1], time[i - 1] - Sfix[-1][0], x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1
    # add last fixation end (we can lose it if dist > maxdist is false for the last point)
    if len(Sfix) > len(Efix):
        Efix.append([Sfix[-1][0], time[len(x) - 1], time[len(x) - 1] - Sfix[-1][0], x[si], y[si]])
    return Sfix, Efix

def get_fixations(eye_positions):
    return np.array(fixation_detection(eye_positions[:, 1], eye_positions[:, 0], eye_positions[:, 2] * 1000.)[1])[:, (3, 4, 0, 1)]

