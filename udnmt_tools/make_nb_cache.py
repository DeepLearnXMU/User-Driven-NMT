import numpy as np
import pickle
import os
import argparse
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from multiprocessing import Pool
from make_cache import Cache, save_cache, chunks
 

TF_THRESHOLD=0.1 # tf-idf score>threshold will pick as keyword   

def tf_idf_keywords(idx, user_doc):
    '''
    Pick key words of user_doc[idx] by tf-idf
    user_doc(list): 
    u0 history+current(all)
    u1 history+current(all)
    ... 
    u_idx history+current[idx](at first) or history+...+current[t] (at now) 
    ...
    '''
    with open('hit_stopwords.txt') as f:
        stlist=f.read().splitlines()
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\u4e00-\u9fa5]+\b', stop_words=stlist)
    X = vectorizer.fit_transform(user_doc)
    tfidf = TfidfTransformer().fit_transform(X)    
    word = vectorizer.get_feature_names()
    nb_keywords=[]
    fa_keywords=[]
    this_user_vector = tfidf[idx]
    cosine_similarities = linear_kernel(this_user_vector, tfidf).flatten()

    max_idx = cosine_similarities.argsort()[-2:-1]
    nb_kid = np.argwhere(tfidf[max_idx]>TF_THRESHOLD)[:,1]
    nb_keywords = np.array(word)[nb_kid]

    # min_idx = cosine_similarities.argsort()[0]
    min_idxs = np.argwhere(cosine_similarities == 0)
    min_idx = min_idxs[random.randint(0, min_idxs.size-1)]
    fa_kid = np.argwhere(tfidf[min_idx]>TF_THRESHOLD)[:,1]
    fa_keywords = np.array(word)[fa_kid]
    return nb_keywords, fa_keywords


def make_cache(uids, user, user_doc, pid, topic_len, nb_cache_file, far_cache_file):
    '''
    uids(list):list of uid
    uder(dict):user dict from json
    user_doc(list):contains all training data(history+current) of all users. len(uids)==len(user_doc)
    pid(int): process id
    ''' 
    print('Run task {}...'.format(os.getpid()))     
    # idx: the index of user_doc, uid: uid key of user dict
    for idx, uid in enumerate(uids):
        nb_cache=Cache(topic_len)
        fa_cache=Cache(topic_len)
        tmp_user_doc=user_doc
        this_user_doc_list=user[uid]['history']
        this_user_doc=' '.join(this_user_doc_list) # for history cache
        print('process user {}'.format(uid))
        for s in user[uid]['current_zh']:  
            this_user_doc=this_user_doc+' '+s
            tmp_user_doc[idx]=this_user_doc
            nb_keywords, fa_keywords = tf_idf_keywords(idx, tmp_user_doc)
            # append keywords to cache
            nb_cache.histoy_update(nb_keywords)
            fa_cache.histoy_update(fa_keywords)

            save_cache(nb_cache, nb_cache_file, pid)
            save_cache(fa_cache, far_cache_file, pid)


def main(args, uid_file, nb_cache_file, far_cache_file):
    # we can only see train and val users when training/validation
    # when testing we can see all users
    with open (uid_file) as uidf:
        uids=uidf.read().splitlines()
    if args.mode=='train':
        with open('data/user/valid.uid') as uidf:
            ap_uids=uidf.read().splitlines()
    elif args.mode=='valid':
        with open('data/user/train.uid') as uidf:
            ap_uids=uidf.read().splitlines()
    elif args.mode=='test':
        with open('data/user/train.uid') as uidf:
            ap_uids=uidf.read().splitlines()
        with open('data/user/valid.uid') as uidf:
            ap_uids+=uidf.read().splitlines()
    full_uids=uids+ap_uids

    with open('data/user/meta.bin', 'rb') as f:
        user=pickle.load(f)

    # make full user doc
    user_doc=[]
    for uid in full_uids:
        user_doc+=[' '.join(user[uid]['history'])+' '.join(user[uid]['current_zh'])]

    # make_cache(uids, user, user_doc)
    l_uids=chunks(uids, args.pnum)
    print('split task to {} processes, each contains {} users'.format(args.pnum, len(l_uids[0])))

    pool = Pool(args.pnum)
    for pid, p_uids in enumerate(l_uids):
        pool.apply_async(make_cache, args=(p_uids, user, user_doc, pid, args.topic_len, nb_cache_file, far_cache_file,))
    pool.close()
    pool.join()
    print('All subprocesses done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find nearby user, far user and save their history cache for every sentence')
    parser.add_argument('--pnum', '-p', type=int, default=1, help='multi-processing number')
    parser.add_argument('--mode', '-m', type=str, choices=['train','valid','test'], help='make cache for train, valid or test set')
    parser.add_argument('--topic_len', '-tl', type=int, default=25, help='topic cache size')
    parser.add_argument('--save_dir', '-s', type=str, help='path to save cache files')

    args = parser.parse_args()

    assert os.path.exists(args.save_dir)
    uid_file='{}/{}.uid'.format(args.save_dir, args.mode)
    nb_cache_file='{}/nearby{}.{}.tmp'.format(args.save_dir, args.topic_len, args.mode)
    far_cache_file='{}/far{}.{}.tmp'.format(args.save_dir, args.topic_len, args.mode)  
    assert not os.path.isfile(args.nb_cache_file)
    assert not os.path.isfile(args.far_cache_file)
    
    main(args, uid_file, nb_cache_file, far_cache_file)

    print('Merging files...')
    # if the following codes cannot work, run in cmd.
    os.popen('for ((i=0;i<{0};i++))do echo {1}/nearby{2}.{3}.tmp.$i;done | xargs -i cat \{\} > {1}/nearby{2}.{3}'.format(args.pnum, args.save_dir, args.topic_len, args.mode))
    os.popen('for ((i=0;i<{0};i++))do echo {1}/far{2}.{3}.tmp.$i;done | xargs -i cat \{\} > {1}/far{2}.{3}'.format(args.pnum, args.save_dir, args.topic_len, args.mode))