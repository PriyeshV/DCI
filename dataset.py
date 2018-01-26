from __future__ import generators, print_function
import numpy as np
from random import shuffle
from scipy.io import loadmat
from copy import deepcopy
import functools
#import Queue
#from multiprocessing import Process, Queue, Manager, Pool
import threading
import time
from os import path
from collections import defaultdict


def async_prefetch_wrapper(iterable, buffer=100):
    """
    wraps an iterater such that it produces items in the background
    uses a bounded queue to limit memory consumption
    """
    done = 'DONE'# object()

    def worker(q, it):
        for item in it:
            q.put(item)
        q.put(done)

    # launch a thread to fetch the items in the background
    queue = Queue.Queue(buffer)

    #pool = Pool()
    #m = Manager()
    #queue = m.Queue()
    it = iter(iterable)
    #workers = pool.apply_async(worker, (queue, it))
    thread = threading.Thread(target=worker, args=(queue, it))
    #thread = Process(target=worker, args=(queue, it))
    thread.daemon = True
    thread.start()
    # pull the items of the queue as requested
    while True:
        item = queue.get()
        if item == 'DONE':#done:
            return
        else:
            yield item

    #pool.close()
    #pool.join()


def async_prefetch(func):
    """
    decorator to make generator functions fetch items in the background
    """
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        return async_prefetch_wrapper(func(*args, **kwds))
    return wrapper

class DataSet(object):
    def __init__(self, cfg):
        """Construct a DataSet.
        """
        self.cfg         = cfg
        self.adj         = self.get_adj(cfg.adjmat_path)
        self.all_labels  = self.get_labels(cfg.label_path)
        self.all_features= self.get_fetaures(cfg.features_path)

        #Increment the positions by 1 and mark the 0th one as False
        self.train_nodes    = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'train_ids.npy'))))
        self.val_nodes      = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'val_ids.npy'))))
        self.test_nodes     = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'test_ids.npy'))))
        # [IMP]Assert no overlap between test/val/train nodes

        self.change = 0
        self.label_cache, self.update_cache = {0:self.all_labels[0]}, {}

        self.wce         = self.get_wce()

    def get_adj(self, path):
        adj = loadmat(path)['adjmat'].toarray()
        # Add dummy '0' nodes
        temp = np.zeros((adj.shape[0] + 1, adj.shape[0] + 1), dtype=int)
        temp[1:, 1:] = adj
        #print('adj: ', np.sum(temp, 0), '\n', np.shape(adj), np.shape(temp))
        return temp

    def get_walks(self, path):
        #Reverse sequences and padding in beginning
        #return np.fliplr(np.loadtxt(path, dtype=np.int))

        walks = np.fliplr(np.loadtxt(path, dtype=np.int))  # reverse the sequence
        seq = deepcopy(walks[:,-1])
        #rotate around the sequences, such that ends are padded with zeros
        for i in range(np.shape(walks)[0]):
            non_zeros = np.sum(walks[i] > 0)
            walks[i] = np.roll(walks[i], non_zeros)
        return walks, seq


    def get_wce(self):
        if self.cfg.solver.wce:
            valid = self.train_nodes #+ self.val_nodes
            tot = np.dot(valid, self.all_labels)
            wce = 1/(len(tot) * (tot*1.0/np.sum(tot)))
        else:
            wce = np.ones(self.all_labels.shape[1])

        wce[np.isinf(wce)] = 0
        print("Cross-Entropy weights: ",wce)
        return wce

    def get_fetaures(self, path):
        # Serves 2 purpose:
        # a) add feature for dummy node 0 a.k.a <EOS> and <unlabeled>
        # b) increments index of all features by 1, thus aligning it with indices in walks
        all_features = np.load(path)
        all_features = all_features.astype(np.float32, copy=False)  # Required conversion for Python3
        all_features = np.concatenate(([np.zeros(all_features.shape[1])], all_features), 0)

        if self.cfg.data_sets.add_degree:
            all_features = np.concatenate((all_features, np.sum(self.adj, axis=0, keepdims=True).T), 1)

        return all_features

    def get_labels(self, path):
        # Labels start with node '0'; Walks_data with node '1'
        # To get corresponding mapping, increment the label node number by 1
        # add label for dummy node 0 a.k.a <EOS> and <unlabeled>
        all_labels = np.load(path)
        all_labels = np.concatenate(([np.zeros(all_labels.shape[1])], all_labels), 0)

        return all_labels

    def get_update_cache(self):
        updated = {}
        for k,v in self.update_cache.items():
            updated[k] = v[0]/v[1]
        return updated

    def accumulate_label_cache(self, labels, nodes):
        #Aggregates all the labels for the corresponding nodes
        #and tracks the count of updates made
        default = (self.all_labels[0], 0) #Initial estimate -> all_zeros

        if self.cfg.data_sets.binary_label_updates:
            #Convert to binary and keep only the maximum value as 1
            amax = np.argmax(labels, axis = 1)
            labels = np.zeros(labels.shape)
            for idx, pos in enumerate(amax):
                labels[idx,pos] = 1
        
        for idx, node in enumerate(nodes):
            prv_label, prv_count = self.update_cache.get(node, default)
            new_label = prv_label + labels[idx]
            new_count = prv_count + 1
            self.update_cache[node] = (new_label, new_count)

    def update_label_cache(self):
        #Average all the predictions made for the corresponding nodes and reset cache
        alpha = self.cfg.solver.label_update_rate

        if len(self.label_cache.items()) <= 1: alpha =1

        for k, v in self.update_cache.items():
            old = self.label_cache.get(k, self.label_cache[0])
            new = (1-alpha)*old + alpha*(v[0]/v[1])
            self.change += np.mean((new - old) **2)
            self.label_cache[k] =  new

        print("\nChange in label: :", np.sqrt(self.change/self.cfg.data_sets._len_vocab)*100)
        self.change = 0
        self.update_cache = {}

    def get_nodes(self, dataset):
        nodes = []
        if dataset == 'train':
            nodes = self.train_nodes
        elif dataset == 'val':
            nodes = self.val_nodes
        elif dataset == 'test':
            nodes = self.test_nodes
        elif dataset == 'all':
            # Get all the nodes except the 0th node
            nodes = [True]*len(self.train_nodes)
            nodes[0] = False
        else:
            raise ValueError

        return nodes

    #@async_prefetch
    def next_batch(self, dataset, batch_size, shuffle=True):

        nodes = np.where(self.get_nodes(dataset))[0]

        # Divide the nodes into buckets based on their number of neighbors
        buckets = 3
        tot_neigh = np.sum(self.adj[nodes], 1)  #get neighbors of the nodes and compute individual sums
        count = zip(nodes, tot_neigh) #zip nodes with their neighbor count
        count = sorted(count, key = lambda item:item[1])
        count = np.array(count)
        buck_size = len(nodes)//buckets
        if len(nodes)%buck_size != 0:
            buckets += 1

        grouped = {}
        for i in range(buckets):
            extra = max(0, (i+1)*buck_size - len(nodes)) #Increase the size of last bucket to accomodate left-over nodes
            temp = count[i*buck_size: (i+1)*buck_size + extra]
            maxi = np.max(temp[:,1])
            grouped[i] = [temp, maxi] #format -> ([..[node, neighbor_count]..], max_neighbor_count)

        if shuffle:
            for k,v in grouped.items():
                indices = np.random.permutation(len(v[0]))
                grouped[k] = [v[0][indices], v[1]]

        tot = buck_size*buckets//batch_size #Total number of batches
        for vertices, maxi in grouped.values():
            #print("Vertices; ",vertices)
            maxi += 1 #number of neighbors + itself
            for idx in range(0, len(vertices), batch_size):
                lengths = []
                #additional dummy entries in the batch to make batch size constant
                dummy = max(0, (idx + batch_size) -len(vertices))
                mask = [1]*batch_size
                if dummy: mask[-dummy:] = [0]*dummy

                #print("mask: ",mask, dummy)
                seq = vertices[idx: idx + batch_size - dummy, 0]
                seq = np.concatenate((seq, [0]*dummy)).astype(int)
                x = []
                for n in seq:
                    x_ = np.where(self.adj[n])[0]
                    np.random.shuffle(x_)  #shuffle neighbor nodes
                    x_ = list(x_)
                    x_.append(n) #append itself to the set of neighbors
                    lengths.append(len(x_))
                    pad = maxi - len(x_) #padding for each sequence
                    x_.extend([0]*pad)
                    #print(list(np.where(self.adj[n])[0]), x_)
                    x.append(x_)

                #print("Shape for this batch: ",np.shape(x))
                x = np.swapaxes(x, 0, 1) #convert from (batch x step) to (step x batch)

                x_labels = [[self.label_cache.get(item, self.all_labels[0]) for item in row] for row in x]
                x_feats = [[self.all_features[item] for item in row] for row in x]
                y = [self.all_labels[item] for item in seq]

                yield (x_feats, x_labels, seq, y, tot, lengths, mask)

    def testPerformance(self):

        start = time.time()
        step =0
        for a,b,c,d,e,f,g in self.next_batch_same('all'):
            step += 1
            if step%500 == 0: print(step)

        print ('total time: ', time.time()-start)