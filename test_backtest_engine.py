import unittest
import pandas as pd
from backtest_engine import simulate_deal, signal_history

def frame(rows):
    return pd.DataFrame(rows, columns=['Open', 'High', 'Low', 'Close'],
                        index=pd.date_range('2026-01-01', periods=len(rows), freq='h')).assign(
                            ATR=1.0, Buy_Signal=False, Sell_Signal=False)

class ExecutionTests(unittest.TestCase):
    def deal(self, df, side=1, hold=35):
        return simulate_deal(df, df.index[0], side, .75, 4, hold, 0, 0)

    def test_next_open_entry(self):
        df = frame([(100,100,100,100),(102,102,102,102)])
        self.assertEqual(self.deal(df)['ENTRY'],102)

    def test_long_gap_stop(self):
        df = frame([(100,100,100,100),(100,100,100,100),(90,91,89,90)])
        self.assertEqual(self.deal(df)['EXIT'],90)
        self.assertEqual(self.deal(df)['PNL %'],-10)

    def test_short_gap_stop(self):
        df = frame([(100,100,100,100),(100,100,100,100),(110,111,109,110)])
        self.assertEqual(self.deal(df,-1)['EXIT'],110)

    def test_ambiguous_bar_stop_first(self):
        df = frame([(100,100,100,100),(100,101,95,100)])
        self.assertEqual(self.deal(df)['EXIT'],96)

    def test_latest_signal_pending(self):
        df = frame([(100,100,100,100)])
        self.assertEqual(self.deal(df)['OUTCOME'],'PENDING')

    def test_timeout_loss_retained(self):
        df = frame([(100,100,100,100),(100,100,98,98)])
        self.assertEqual(self.deal(df,hold=1)['OUTCOME'],'TIME-LOSS')

    def test_open_loss_marked(self):
        df = frame([(100,100,100,100),(100,100,98,98)])
        rec = self.deal(df)
        self.assertEqual(rec['OUTCOME'],'OPEN')
        self.assertEqual(rec['PNL %'],-2)

    def test_no_overlapping_positions(self):
        df = frame([(100,100,100,100)]*5)
        df['Buy_Signal'] = True
        self.assertEqual(len(signal_history(df,.75,4)),1)

    def test_costs_reduce_return(self):
        df = frame([(100,100,100,100),(100,100,100,100)])
        rec = simulate_deal(df,df.index[0],1,.75,4,max_hold=1)
        self.assertLess(rec['PNL %'],0)

if __name__ == '__main__':
    unittest.main()
