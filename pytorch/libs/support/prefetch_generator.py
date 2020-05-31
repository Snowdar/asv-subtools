# -*- coding: utf-8 -*-

# Github: https://github.com/justheuristic/prefetch_generator.

"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
This is a single-function package that transforms arbitrary generator into a background-thead generator that prefetches several batches of data in a parallel background thead.
This is useful if you have a computationally heavy process (CPU or GPU) that iteratively processes minibatches from the generator while the generator consumes some other resource (disk IO / loading from database / more CPU if you have unused cores). 
By default these two processes will constantly wait for one another to finish. If you make generator work in prefetch mode (see examples below), they will work in parallel, potentially saving you your GPU time.
We personally use the prefetch generator when iterating minibatches of data for deep learning with tensorflow and theano ( lasagne, blocks, raw, etc.).
Quick usage example (ipython notebook) - https://github.com/justheuristic/prefetch_generator/blob/master/example.ipynb
This package contains two objects
 - BackgroundGenerator(any_other_generator[,max_prefetch = something])
 - @background([max_prefetch=somethind]) decorator
the usage is either
#for batch in BackgroundGenerator(my_minibatch_iterator):
#    doit()
or
#@background()
#def iterate_minibatches(some_param):
#    while True:
#        X = read_heavy_file()
#        X = do_helluva_math(X)
#        y = wget_from_pornhub()
#        do_pretty_much_anything()
#        yield X_batch, y_batch
More details are written in the BackgroundGenerator doc
help(BackgroundGenerator)
"""



import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)

#decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch
    def __call__(self, gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs), max_prefetch=self.max_prefetch)
        return bg_generator