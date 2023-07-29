from matplotlib import pyplot as plt
import numpy as np

################################# Parameters #################################
# Sound parameters and calculations
speed_of_sound = 34314 #cm/s
rate = 44100

# Microphone positions in cm
microphone2 = np.array([8,-16])
microphone3 = np.array([-8,-16])

# Technical parameters
window_size = 1000
min_distance = 25     # twice the furthest distance between microphones     
max_distance = 200    # 2 meters

################################# Functions ##################################
def find_max_delay(microphone2, microphone3, rate = 44100, speed_of_sound = 34300):
    """Finds the maximum delay between the two microphones
    input:
        microphone2 (np.array): the position of the second microphone
        microphone3 (np.array): the position of the third microphone
        rate (int): the rate of the sound
        speed_of_sound (float): the speed of sound in cm/s
        
    output:
        max_delay (int): the maximum delay between the two microphones"""
    
    # find the largest distance between the microphones or the origin 
    largest_distance = np.max([np.linalg.norm(microphone2), np.linalg.norm(microphone3), np.linalg.norm(microphone2 - microphone3)])
    max_delay = int(np.floor(rate * largest_distance / speed_of_sound))
    return max_delay

def sliding_optimize(sound1, sound2, max_delay, window_size=5000):
    """Finds the delay between two sounds using a sliding optimization method
    inpute:
        sound1 (np.array): the first sound
        sound2 (np.array): the second sound
        window_size (int): the size of the window to use for the optimization
    
    output:
        delay (int): the delay between the two sounds"""

    # Get the last window_size samples
    sound1 = sound1[-window_size:]
    sound2 = sound2[-window_size:]
    
    # Initialize the scores and delay values arrays and get the length of the sounds
    delay_vals = np.arange(-max_delay+1, max_delay)
    scores = np.empty(delay_vals.shape)

    # Loop through the possible delay values and trim the sounds accordingly
    for i, delay in enumerate(delay_vals):
        if delay > 0:
            test1 = sound1[delay:]
            test2 = sound2[:-delay]

        # If the delay is negative, account for python's negative indexing
        elif delay < 0:
            test1 = sound1[:delay]
            test2 = sound2[-delay:]

        # If the delay is 0, just use the sounds as is
        else:
            test1 = sound1
            test2 = sound2
        
        # Calculate the difference between the two sounds and normalize it
        scale = len(test1)
        scores[i] = np.linalg.norm(test1 - test2, 2) / scale

    # Find the argmin of the scores to see which delay is most alligned, and return the delay value
    return delay_vals[np.argmin(scores)]


def find_directions(point, c, eps = .01):
    """Finds the two directions that the v-shape can take
    input:
        point (np.array): the location of the microphone relative to the first one
        c (float): the change in time between the two microphones
        eps (float): the tolerance for the vertical line

    output:
        direction1 (np.array): the first direction
        direction2 (np.array): the second direction"""

    # get the rotation matrix to rotate the line to the x axis
    point = np.array(point)
    angle = np.arctan2(point[1], point[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

    # get the distance from the origin and the starting point
    d = np.linalg.norm(point)
    
    # If the line is vertical, choose the two basis vectors accordingly
    if np.abs(c)/d < eps:
        direction1 = np.array([0, 1])
        direction2 = np.array([0, -1])

    # If the second point is later than the first, then the v-shape is going to hug the first point
    elif c > 0:
        slope = np.sqrt((d**2 / c**2) - 1)
        direction1 = -np.array([1, -slope])
        direction2 = -np.array([1, slope])

    # If the second point is earlier than the first, then the v-shape is going to hug the second point
    else:
        slope = np.sqrt((d**2 / c**2) - 1)
        direction1 = np.array([1, slope])
        direction2 = np.array([1, -slope])
    
    # Rotate the vectors and normalize them
    direction1 = (rot @ direction1)
    direction2 = (rot @ direction2)
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)
    
    # Return the two directions
    return np.round(direction1,4), np.round(direction2,4)

def solve_intersect(direction1, direction2, midpoint1, midpoint2, eps = .01):
    """Solves for the intersection of two lines with only positive scalars
    input:
        direction1 (np.array): the first direction
        direction2 (np.array): the second direction
        midpoint1 (np.array): the first midpoint
        midpoint2 (np.array): the second midpoint
        eps (float): the tolerance for the vertical line

    output:
        location (np.array): the location of the intersection"""
    
    # Make the direction matrix and check if lines are parallel
    D = np.array([direction1,direction2]).T
    if np.linalg.norm(direction1 - direction2) < eps:
        return np.array([np.inf,np.inf])
    
    # return nothing if the lines are parallel but in opposite directions
    if np.linalg.det(D) < eps:
        return None

    # Solve for the scalers and negate the second scaler to solve for equation
    scalers = np.linalg.inv(D) @ (midpoint2 - midpoint1)
    scalers[1] *= -1

    # Find the location and return it if both scalars are positive
    if scalers[0] > 0 and scalers[1] > 0:
        return (D @ scalers + midpoint1 + midpoint2) / 2

def triangulate(p1,p2, t1, t2, mind_dist = 3, max_dist = 100, eps = .01):
    """Finds the location of the sound source using the three microphones
    input:
        p1 (np.array): the position of the first microphone
        p2 (np.array): the position of the second microphone
        t1 (float): the time difference between the first and second microphone
        t2 (float): the time difference between the first and third microphone
        mind_dist (float): the minimum distance between the microphones and the sound source
        max_dist (float): the maximum distance between the microphones and the sound source
        
    output:
        location (np.array): the location of the sound source"""

    # Get the 4 possible directions and make a direction array
    d11,d12 = find_directions(p1,t1, eps = eps)
    d21,d22 = find_directions(p2,t2, eps = eps)


    # Get the 2 midpoints
    midpoint1 = p1 / 2
    midpoint2 = p2 / 2

    # Solve for the 4 possible locations
    location1 = solve_intersect(d11,d22,midpoint1,midpoint2, eps = eps)
    location2 = solve_intersect(d11,d21,midpoint1,midpoint2, eps = eps)
    location3 = solve_intersect(d12,d22,midpoint1,midpoint2, eps = eps)
    location4 = solve_intersect(d12,d21,midpoint1,midpoint2, eps = eps)

    # set None valued locations to the origin
    if location1 is None:
        location1 = np.array([0,0])
    if location2 is None:
        location2 = np.array([0,0])
    if location3 is None:
        location3 = np.array([0,0])
    if location4 is None:
        location4 = np.array([0,0])
    
    # make an array of the locations and get max index
    locations = np.array([location1, location2, location3, location4])
    distances = np.linalg.norm(locations, axis = 1)
    max_index = np.argmax(distances)
    
    # if the furthest distances is larger than max_dist, return the cut off location
    if distances[max_index] > max_dist:
        # if the max_index is 0 or 3, return the sum of the two directions scaled by max_dist
        if max_index == 0 or max_index == 3:
            final_direction = (d11 + d22)
            final_direction = final_direction / np.linalg.norm(final_direction)
            return final_direction * max_dist
        # if the max_index is 1 or 2, return the sum of the two directions scaled by max_dist
        else:
            final_direction = (d12 + d21)
            final_direction = final_direction / np.linalg.norm(final_direction)
            return final_direction * max_dist
    
    # if the furthest distance is smaller than mind_dist, return None
    elif distances[max_index] < mind_dist:
        return None
    
    # otherwise return the location
    else:
        return locations[max_index]
    
def radar(microphone2, microphone3, sound1, sound2, sound3, min_dist = 3, max_dist = 100, window_size = 5000, rate = 44100, speed_of_sound = 34314, eps = .01):
    """Finds the location of the loudest sound using the recordings from the three microphones
    input: 
        microphone2 (np.array): the position of the second microphone
        microphone3 (np.array): the position of the third microphone
        sound1 (np.array): the first sound (make its size be at least window_size + max_delay + 1)
        sound2 (np.array): the second sound (make its size be at least window_size + max_delay + 1)
        sound3 (np.array): the third sound (make its size be at least window_size + max_delay + 1)
        min_dist (float): the minimum distance between the microphones and the sound source
        max_dist (float): the maximum distance between the microphones and the sound source
        window_size (int): the size of the window to use for the optimization
    
    output:
        location (np.array): the location of the loudest sound source"""
    
    # find the max delay between the two microphones
    max_delay = find_max_delay(microphone2, microphone3, rate, speed_of_sound)

    # Find the delays between the sounds
    delay2 = sliding_optimize(sound1, sound2, max_delay, window_size)
    delay3 = sliding_optimize(sound1, sound3, max_delay, window_size)

    # convert the delay into a distance
    delay2 = delay2 * speed_of_sound / rate
    delay3 = delay3 * speed_of_sound / rate

    # handle edge cases where values are small
    if delay2 == delay3:
        if delay2 < 0:
            return np.array([0, -max_dist])
        elif delay2 > 0:
            return np.array([0, max_dist])
        
    # Find the location of the sound source
    location = triangulate(microphone2, microphone3, delay2, delay3, min_dist, max_dist, eps = eps)
    
    # Return the location
    return location


################################# Run Simulations ##################################
# Note, it may have the delay be negative
radar()