import numpy as np
import pickle
import os
import math
import argparse
from queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from multiprocessing import Pool

 
TF_THRESHOLD=0.1 #tf-idf score>threshold will pick as keyword

class UniqQueue(Queue):    
    def put(self, item, block=True, timeout=None, unique=False):
        with self.not_full:
            if unique:
                if item in self.queue:
                    return
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class Cache(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.content = UniqQueue(max_len)
        self.nb_word_num = 0

    def length(self):
        return self.content.qsize()

    def _isfull(self):
        return self.length()==self.max_len
    
    def get(self):
        return self.content.get()

    def put(self, x):
        if self._isfull():
            self.get()
        self.content.put(x, block=False, unique=True)  
      
    def fifo_update(self, keywords, init_history=False):
        '''
        first-in-fist-out 
        '''
        for kw in keywords:         
            self.put(kw)
        if not init_history:
            self.nb_word_num-=1
    
    def histoy_update(self, keywords, init_history=False):        
        if init_history:
            self.nb_word_num = len(keywords)           
        elif self.nb_word_num < 0:
            # no nb words, totally update
            for i in range(self.length()):
                self.get()
        self.fifo_update(keywords, init_history=init_history)           


def tf_idf_keywords(idx, user_doc, find_nearby=False):
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
    kid = np.argwhere(tfidf[idx]>TF_THRESHOLD)[:,1]
    word = vectorizer.get_feature_names()
    keywords = np.array(word)[kid]
    
    nb_keywords=[]
    if find_nearby:
        this_user_vector = tfidf[idx]
        cosine_similarities = linear_kernel(this_user_vector, tfidf).flatten()
        max_idx = cosine_similarities.argsort()[-2:-1]
        nb_kid = np.argwhere(tfidf[max_idx]>TF_THRESHOLD)[:,1]
        nb_keywords = np.array(word)[nb_kid]
    return keywords, nb_keywords


def save_cache(cache, filename, pid):
    '''
    get the item from cache, append to file, put the item back
    cache(class Cache) 
    pid(int):process id
    '''
    filename=filename+'.'+str(pid)
    with open(filename,'a') as f:
        for idx in range(cache.length()):
            w=cache.get()
            f.write(w+' ')
            cache.put(w)
        f.write('\n')


def make_cache(uids, user, user_doc, pid, topic_len, context_len, topic_cache_file, context_cache_file):
    '''
    uids(list):list of uid
    uder(dict):user dict from json
    user_doc(list):contains all training data(history+current) of all users. len(uids)==len(user_doc)
    pid(int): process id
    ''' 
    print('Run task {}...'.format(os.getpid()))     
    # idx: the index of user_doc, uid: uid key of user dict
    for idx, uid in enumerate(uids):
        topic_cache=Cache(topic_len)
        context_cache=Cache(context_len)
        tmp_user_doc=user_doc
        this_user_doc_list=user[uid]['history'] # for context cache
        this_user_doc=' '.join(this_user_doc_list) # for topic cache
        print('process user {}'.format(uid))
        for s in user[uid]['current_zh']:
            this_user_doc_list.append(s)
            cur_keywords, _ = tf_idf_keywords(len(this_user_doc_list)-1, this_user_doc_list)
            # print ('current keywords: {}'.format(cur_keywords))
            context_cache.fifo_update(cur_keywords)
            save_cache(context_cache, context_cache_file, pid)
            
            this_user_doc=this_user_doc+' '+s
            tmp_user_doc[idx]=this_user_doc
            print('len', len(this_user_doc_list))
            if len(this_user_doc_list)==1:
                # first current sentence and no history
                # init topic_cache with nearby user's
                his_keywords, nb_keywords = tf_idf_keywords(idx, tmp_user_doc, find_nearby=True)
                topic_cache.histoy_update(nb_keywords, init_history=True)
                topic_cache.histoy_update(his_keywords)
            else:
                his_keywords, _ = tf_idf_keywords(idx, tmp_user_doc)
            # append keywords to cache
            topic_cache.histoy_update(his_keywords)
            save_cache(topic_cache, topic_cache_file, pid)


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def main(args, uid_file, topic_cache_file, context_cache_file):
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
        pool.apply_async(make_cache, args=(p_uids, user, user_doc, pid, args.topic_len, args.context_len, topic_cache_file, context_cache_file,))
    pool.close()
    pool.join()
    print('All subprocesses done.')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make topic cache and context cache for every sentence')
    parser.add_argument('--pnum', '-p', type=int, default=1, help='multi-processing number')
    parser.add_argument('--mode', '-m', type=str, choices=['train','valid','test'], help='make cache for train, valid or test set')
    parser.add_argument('--topic_len', '-tl', type=int, default=25, help='topic cache size')
    parser.add_argument('--context_len', '-cl', type=int, default=35, help='context cache size')
    parser.add_argument('--save_dir', '-s', type=str, help='path to save cache files')

    args = parser.parse_args()

    assert os.path.exists(args.save_dir)
    uid_file='{}/{}.uid'.format(args.save_dir, args.mode)
    topic_cache_file='{}/topic{}.{}.tmp'.format(args.save_dir, args.topic_len, args.mode)
    context_cache_file='{}/context{}.{}.tmp'.format(args.save_dir, args.context_len, args.mode)  
    assert not os.path.isfile(args.topic_cache_file)
    assert not os.path.isfile(args.context_cache_file)
    
    main(args, uid_file, topic_cache_file, context_cache_file)

    print('Merging files...')
    # if the following codes cannot work, run in cmd.
    os.popen('for ((i=0;i<{0};i++))do echo {1}/topic{2}.{3}.tmp.$i;done | xargs -i cat \{\} > {1}/topic{2}.{3}'.format(args.pnum, args.save_dir, args.topic_len, args.mode))
    os.popen('for ((i=0;i<{0};i++))do echo {1}/context{2}.{3}.tmp.$i;done | xargs -i cat \{\} > {1}/context{2}.{3}'.format(args.pnum, args.save_dir, args.context_len, args.mode))