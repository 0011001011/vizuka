import pandas as pd

class Annotations(pd.DataFrame):
    def __init__(self, array):
        columns = [
                'pk',
                'amount',
                'operation_code',
                'operation_date',
                'bank_wording',
                'creditor_id',
                'bank_code',
                'activity_start_date',
                'ape_code',
                'vat_regime',
                'fiscal_regime',
                'imposition_regime',
                'bnc_option',
                'legal_obligation',
                'libellé',
                'account',
                'vat_type',
                'vat_intra_flag',
                'vat_class'
                ]
        super(Annotations, self).__init__(array, columns=columns)
