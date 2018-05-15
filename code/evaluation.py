"""
Helper methods for evaluating the accuracy of our prediction.
"""
#Accuracy = Pr(truth=pred)
#should be computed as:
     #sum(truth == pred)/len(E)
#Precision = Pr(pred|truth)
#Computed as:
     #sum(truth and pred)/ sum(truth)
#recall = pr(truth|pred)
#computed as:
     #sum(truth and pred) / sum(pred)
import numpy as np
from helper_routines import convolve_and_score

def classifier(bounds, width, rho, sigma_x):
    """
    takes:
    bounds: [width, height]
    width: float, width of a box
    rho: (np.array(n_points, 2), np.array(n_points))
    tau: float
    returns: float array
    """
    print(bounds)
    ctx = int(np.ceil(bounds[0]/width))
    cty = int(np.ceil(bounds[1]/width))


    bboxes = []
    sums = []
    for i in range(ctx):
        for j in range(cty):
            bboxes.append([-1 * bounds[0]/2.0 + width * i, -1 * bounds[0]/2.0 + width * (i+1),
                           -1 * bounds[1]/2.0 + width * j, -1 * bounds[1]/2.0 + width * (j+1)])
    bboxes = np.array(bboxes)
    if sigma_x > 0:
        sums = convolve_and_score(rho[0], rho[1], sigma_x, bboxes)
    else:
        sums, bboxes = classifier_no_convolve(bounds, width, rho)

    #for i in range(ctx):
    #    lbx = x_lims[i]
    #    ubx = x_lims[i+1]
    #    start = pts[0].searchsorted(lbx)
    #    end = pts[0].searchsorted(ubx)
    #    pts_x = pts[:,start:end]
    #    weights_x = weights[start:end]


    #    #Sort with respect to y-component
    #    indices = np.argsort(pts_x[1])
    #    pts_x = pts_x[:,indices]
    #    weights_x = weights_x[indices]
    #    for j in range(cty):
    #        lby = y_lims[j]
    #        uby = y_lims[j+1]
    #        start = pts_x[1].searchsorted(lby)
    #        end = pts_x[1].searchsorted(uby)
    #        sums.append(weights_x[start:end].sum())


    sums = np.array(sums)
    return sums, bboxes

def classifier_no_convolve(bounds, width, rho):
    """
    takes:
    bounds: [width, height]
    rho: (np.array(n_points, 2), np.array(n_points))
    tau: float
    returns: float array #NOTE hoj:  This does not output a bool, as written
    """
    pts, weights = rho

    indices = np.argsort(pts[0])
    pts = pts[:, indices]
    weights = weights[indices]

    ctx = int(np.ceil(bounds[0]/width))
    cty = int(np.ceil(bounds[1]/width))
    x_lims = np.linspace(-bounds[0]/2, bounds[0]/2, ctx+1)
    y_lims = np.linspace(-bounds[1]/2, bounds[1]/2, cty+1)

    bboxes = []
    sums = []

    for i in range(ctx):
        lbx = x_lims[i]
        ubx = x_lims[i+1]
        start = pts[0].searchsorted(lbx)
        end = pts[0].searchsorted(ubx)
        pts_x = pts[:, start:end]
        weights_x = weights[start:end]


        #Sort with respect to y-component
        indices = np.argsort(pts_x[1])
        pts_x = pts_x[:, indices]
        weights_x = weights_x[indices]
        for j in range(cty):
            lby = y_lims[j]
            uby = y_lims[j+1]
            start = pts_x[1].searchsorted(lby)
            end = pts_x[1].searchsorted(uby)
            sums.append(weights_x[start:end].sum())
            bboxes.append([-1 * bounds[0]/2.0 + width * i, -1 * bounds[0]/2.0 + width * (i+1),
                           -1 * bounds[1]/2.0 + width * j, -1 * bounds[1]/2.0 + width * (j+1)])

    #print(res + integrals)[np.where(res + integrals > tau)[0]][36:]
    #printE[np.where(res + integrals > tau)[0]]
    sums = np.array(sums)
    bboxes = np.array(bboxes)
    return sums, bboxes



def true_classifier(E, rho_true):
    """
    takes:
    E: [[[box_width, box_height], [box_x, box_y]]]
    rho_true: function
    tau: float
    returns: bool array
    """
    E = np.array(E)
    integrals = rho_true(E)
    #print E[np.where(integrals > tau)[0]]
    return integrals


def precision(pred, truth):
    """
    Computes precisions.
    takes:
    pred: bool array
    truth: bool array
    returns:
    precision: float
    """
    if float(len(np.where(pred)[0])) != 0:
        return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(pred)[0]))
    return 1

def recall(pred, truth):
    """
    Computes precisions.
    takes:
    pred: bool array
    truth: bool array
    returns:
    recall: float
    """
    if float(len(np.where(truth)[0])) != 0:
        return len(np.where(np.logical_and(pred, truth))[0]) / float(len(np.where(truth)[0]))
    return 1

def accuracy(pred, truth):
    """
    Computes precisions.
    takes:
    pred: bool array
    truth: bool array
    returns:
    accurayc: float
    """
    return len(np.where(pred == truth)[0]) / float(len(truth))

def evaluate_ours(bbox, rho, rho_lin, rho_true, width, sigma):
    """
    Takes:
    bbox: np.array(2): [scene_width, scene_height]
    rho: (np.array(n_points, 2), np.array(n_points))
    rho_true: function
    tau: float
    lin_term: function
    width: float
    returns (precision, recall, accuracy)
    """

    bboxes = []
    #create all of the bounding boxes
    #Create prediction
    predic, bboxes = classifier(bbox, width, rho, sigma)
    predicl, _ = classifier_no_convolve(bbox, width, rho_lin)
    predic = predic + predicl
    #create truth for comparison
    truth = true_classifier(bboxes, rho_true)
    true = truth > 0

    return  (predic, true, bboxes)

def evaluate_lin(bbox, rho_lin, rho_true, width):
    """
    Takes:
    bbox: np.array(2): [scene_width, scene_height]
    rho: (np.array(n_points, 2), np.array(n_points))
    rho_true: function
    tau: float
    lin_term: function
    width: float
    returns (precision, recall, accuracy)
    """

    bboxes = []
    #create all of the bounding boxes
    #Create prediction
    predic, bboxes = classifier_no_convolve(bbox, width, rho_lin)
    #create truth for comparison
    truth = true_classifier(bboxes, rho_true)
    true = truth > 0

    return  (predic, true)



if __name__ == "__main__":
    rand1 = np.random.random([1000]) > 0.5
    print("Precision test 1:")
    print(precision(rand1, rand1))
    print("Recall test 1:")
    print(recall(rand1, rand1))
    print("Accuracy test 1:")
    print(accuracy(rand1, rand1))

    #test 4 bounding box exhaustively
    #it = itertools.product("01", repeat = 8)
    #for i in it:
    #    fn = lambda x: x == "1"
    #    b1 = map(fn, i[0:4])
    #    b2 = map(fn, i[4:])
    #    try:
    #        print precision(b1, b2)
    #    except:
    #        pass
    #    try:
    #        print recall(b1, b2)
    #    except:
    #        pass
    #    try:
    #        print accuracy(b1, b2)
    #    except:
    #        pass


    #Test dirac deltas
    rho = (np.array([[0.5, 0.5], [-0.5, -0.5]]), np.array([0.2, 0.1]))
    bounds = (1, 1)
    width = 0.5

    def fn(x, y):
        x = x.flatten()
        return np.zeros(len(x))
    print("should return [True, False, True, False]:")
    print(classifier(E, rho, fn))

    assert False

    bbox = np.array([0.4, 0.4])
    tau = 0.1
    rho = (np.array([[-0.1, 0.1],
                     [0.11, 0.11],
                     [0, 0],
                     [-.1, -.1],
                     [-.1, 0]]),
           np.array([0.2, 0.2, 0.2, .2, .2]))

    lin_term = lambda x, y: np.zeros(len(x.flatten()))

    def rho_true(bounding_boxes):
        """
        Auxiliary test function that models a distribution where there is no pedestrian
        """
        return np.zeros([len(bounding_boxes)])

    resolution = [2, 2]

    #print(evaluate_plane(bbox, rho, rho_true, tau, lin_term, resolution))
