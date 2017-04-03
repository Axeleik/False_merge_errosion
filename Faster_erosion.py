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
    segmented = np.array(g["z/1/test"])             #noch nicht np.uint8


    Correct = []
    False_merges = []
    Zwischenspeicher = []
    False_splits = []
    Zu_klein_seg = []
    Zu_klein_gt =[]

    a=len(np.unique(segmented))
    print "Es gibt ", len(np.unique(segmented))," Segmente"




    seg_general_obj_mask_z = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmented[:, :, z])[:, :, None] for z in
         xrange(segmented.shape[2])],
        axis=2
    )

    seg_general_obj_mask_y = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmented[:, y, :])[:, None, :] for y in
         xrange(segmented.shape[1])],
        axis=1
    )
    seg_general_obj_mask = copy.deepcopy(segmented)
    seg_general_obj_mask[seg_general_obj_mask_y == 1] = 0
    seg_general_obj_mask[seg_general_obj_mask_z == 1] = 0
    seg_eroded_general_obj_mask = vigra.filters.discErosion(seg_general_obj_mask.astype(np.uint8), 10)

    # FIXME hack to make the discErosion capable of uint>8
    seg_general_obj_mask[seg_eroded_general_obj_mask == 0] = 0

    gt_general_obj_mask_z = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmented[:, :, z])[:, :, None] for z in
         xrange(segmented.shape[2])],
        axis=2
    )

    gt_general_obj_mask_y = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmented[:, y, :])[:, None, :] for y in
         xrange(segmented.shape[1])],
        axis=1
    )
    gt_general_obj_mask = copy.deepcopy(ground_truth)
    gt_general_obj_mask[gt_general_obj_mask_y == 1] = 0
    gt_general_obj_mask[gt_general_obj_mask_z == 1] = 0
    gt_eroded_general_obj_mask = vigra.filters.discErosion(gt_general_obj_mask.astype(np.uint8), 10)

    gt_general_obj_mask[gt_eroded_general_obj_mask == 0] = 0

    print "Es gibt2 ", len(np.unique(seg_general_obj_mask)), " Segmente"


    for i in np.unique(seg_general_obj_mask):

        print "\nnummer ",i,":"

        if len(np.unique(ground_truth[seg_general_obj_mask==i]))>1:

            print "False merge"
            False_merges.append(i)

    #    elif len(seg_general_obj_mask[seg_general_obj_mask == i]) == 0:
    #        Zu_klein_seg.append(i)
    #        print "Zu klein bei Segmentierung "

        else:
            Zwischenspeicher.append(ground_truth[np.where(seg_general_obj_mask==i)[0][0],np.where(seg_general_obj_mask==i)[1][0],np.where(seg_general_obj_mask==i)[2][0]])
            print "Zwischenspeicher"

    Zwischenspeicher_backup=copy.deepcopy(Zwischenspeicher)
    print "Nun die zweite schleife "
    print "Zwischenspeicher hat " , len(Zwischenspeicher)," objekte"
    obj_nummer=0
    for i in Zwischenspeicher:
        print "\nnummer ", obj_nummer, ":"

        if len(np.unique(segmented[gt_general_obj_mask==i]))>1:
            print "False Split"

            False_splits.append((np.unique(segmented[gt_general_obj_mask==i])))

    #    elif len(np.unique([gt_general_obj_mask == i])) == 0:
    #        Zu_klein_gt.append(i)
    #        print "Zu klein bei Ground truth "

        else:
            print "Passende Segmentierung"
            Correct.append(np.unique(segmented[gt_general_obj_mask==i]))
        obj_nummer=obj_nummer+1




    print "\nCorrect:", Correct
    with open("/media/axeleik/EA62ECB562EC8821/data/Correct.pkl", mode='w') as f:
        pickle.dump(Correct, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/Correct.txt", Correct)

    print "\nZwischenspeicher:", Zwischenspeicher
    with open("/media/axeleik/EA62ECB562EC8821/data/Zwischenspeicher.pkl", mode='w') as f:
        pickle.dump(Zwischenspeicher, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/Zwischenspeicher.txt",Zwischenspeicher)


    print "\nFalse_merges:", False_merges
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(False_merges, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/False_merges.txt", False_merges)

    print "\nZu klein bei Segmentierung:", Zu_klein_seg
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(Zu_klein_seg, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/False_merges.txt", Zu_klein_seg)

    print "\nZu klein bei Ground truth:", Zu_klein_gt
    with open("/media/axeleik/EA62ECB562EC8821/data/False_merges.pkl", mode='w') as f:
        pickle.dump(Zu_klein_gt, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/False_merges.txt", Zu_klein_gt)

    print "\nIm Zwischenspeicher Backup:", Zwischenspeicher_backup
    with open("/media/axeleik/EA62ECB562EC8821/data/Zwischenspeicher_Backup.pkl", mode='w') as f:
        pickle.dump(Zwischenspeicher_backup, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/Zwischenspeicher_Backup.txt", Zwischenspeicher_backup)

    print "\nFalse splits:", False_splits
    with open("/media/axeleik/EA62ECB562EC8821/data/False_splits.pkl", mode='w') as f:
        pickle.dump(False_splits, f)
    np.savetxt("/media/axeleik/EA62ECB562EC8821/data/False_splits.txt", False_splits)







