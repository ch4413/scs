###
from scsavailability import features as feat


class ScsData():
    def __init__(self, name, av, at, fa):
        self.name = name
        self.av = av
        self.at = at
        self.fa = fa
        self.av_processed = []
        self.at_processed = []
        self.fa_processed = []
        self.unmapped = []
        self.df = []

    def pre_process_at(self):
        self.at_processed = feat.pre_process_at(self.at)

    def pre_process_av(self):
        self.av_processed = feat.pre_process_av(self.av)

    def pre_process_faults(self):
        self.fa_processed, self.unmapped = feat.pre_process_fa(self.fa)

    def floor_shift_time_fa(self, shift=0):
        self.fa_processed = feat.floor_shift_time_fa(self.fa_processed,
                                                     shift=shift)
        print('fa_processed updated with %d min time shift' %shift)

    def create_ptt_df(self, weights):
        self.df, fa_ptt = feat.create_PTT_df(self.fa_processed,
                                             self.at_processed,
                                             self.av_processed,
                                             weights=weights)
        return fa_ptt

    def log_totes(self):
        self.df = feat.log_totes(self.df)
        print('df edited: log transformation of totes')
