import numpy as np

def get_ROI(subj, e, brain_ROI, task = None):
    '''
    Returns an ROI for a given subj and elec.
    If subject is GP35 then must give a task argument so knows which ROIs to pull
    '''
    
    if (subj == 'GP35') & ((task == 'EmoRep') | (task == 'EmoGen')):
        subj = 'GP35_words'
    elif (subj == 'GP35') & ((task == 'FaceEmo') | (task == 'FaceGen')):
        subj = 'GP35_face'
    
    try:
        rois = brain_ROI[subj]
    except:
        return None
    else:
        for roi, elecs in rois.iteritems():
            if np.in1d(e, elecs):
                return roi