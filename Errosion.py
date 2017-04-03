import h5py
import numpy as np
import vigra.filters as fil
import vigra
import copy
import pickle

def printname(name):
    print name

if __name__ == "__main__":
    pass
    f = h5py.File("/media/axeleik/EA62ECB562EC8821/data/ground_truth.h5", mode='r')
    g = h5py.File("/media/axeleik/EA62ECB562EC8821/data/result_resolved.h5", mode='r')
    g.visit(printname)

    ground_truth = np.array(f["z/1/neuron_ids"])    #noch nicht np.uint8
    segmented = np.array(g["z/1/test"])#noch nicht np.uint8


    Correct = []
    False_merges = []
    Zwischenspeicher = []
    False_splits = []
    Zu_klein_seg = []
    Zu_klein_gt =[]

    a=len(np.unique(segmented))
    print "Es gibt ", len(np.unique(segmented))," Segmente"

    for i in np.unique(segmented):
        uniq=np.array(copy.deepcopy(segmented))
        uniq[segmented!=i]=0
        uniq[segmented == i] = 1
        print "\nnummer ",i,":"
        uniq=fil.discErosion(np.array(uniq,np.uint8),15)

        if len(np.unique(ground_truth[uniq==1]))>1:

            print "False merge"
            False_merges.append(i)

        elif len(uniq[uniq == 1]) == 0:
            Zu_klein_seg.append(i)
            print "Zu klein bei Segmentierung "

        else:
            Zwischenspeicher.append(ground_truth[np.where(uniq==1)[0][0],np.where(uniq==1)[1][0],np.where(uniq==1)[2][0]])
            print "Zwischenspeicher"

    print "Nun die zweite schleife "
    print "Zwischenspeicher hat " , len(Zwischenspeicher)," objekte"
    obj_nummer=0
    for i in Zwischenspeicher:
        print "\nnummer ", obj_nummer, ":"
        uniq = np.array(copy.deepcopy(ground_truth))
        uniq[ground_truth != i] = 0
        uniq[ground_truth == i] = 1
        uniq = fil.discErosion(np.array(uniq, np.uint8), 15)

        if len(np.unique(segmented[uniq==1]))>1:
            print "False Split"

            False_splits.append((np.unique(segmented[uniq==1])))

        elif len(uniq[uniq == 1]) == 0:
            Zu_klein_gt.append(i)
            print "Zu klein bei Ground truth "

        else:
            print "Passende Segmentierung"
            Correct.append(np.unique(segmented[uniq==1]))
        obj_nummer=obj_nummer+1

    print "\nCorrect:", Correct
    with open("/media/axeleik/EA62ECB562EC8821/data/Correct.pkl", mode='w') as f:
        pickle.dump(Correct, f)

    print "\nZwischenspeicher:", Zwischenspeicher
    with open("/media/axeleik/EA62ECB562EC8821/data/Zwischenspeicher.pkl", mode='w') as f:
        pickle.dump(Zwischenspeicher, f)

    print "\nFalse splits:", False_splits
    with open("/media/axeleik/EA62ECB562EC8821/data/False_splits.pkl", mode='w') as f:
        pickle.dump(False_splits, f)

    print "\nFalse_merges:", False_merges
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(False_merges, f)

    print "\nZu klein bei Segmentierung:", Zu_klein_seg
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(Zu_klein_seg, f)

    print "\nZu klein bei Ground truth:", Zu_klein_gt
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(Zu_klein_gt, f)




 general_obj_mask_z = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(original_seg[:, :, z])[:, :, None] for z in xrange(original_seg.shape[2])],
        axis=2
    )
    general_obj_mask_y = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(original_seg[:, y, :])[:, None, :] for y in xrange(original_seg.shape[1])],
        axis=1
    )
    general_obj_mask = copy.deepcopy(original_seg)
    general_obj_mask[general_obj_mask_y == 1] = 0
    general_obj_mask[general_obj_mask_z == 1] = 0
    eroded_general_obj_mask = vigra.filters.discErosion(general_obj_mask.astype(np.uint8), 10)
    # FIXME hack to make the discErosion capable of uint>8
    general_obj_mask[eroded_general_obj_mask == 0] = 0



