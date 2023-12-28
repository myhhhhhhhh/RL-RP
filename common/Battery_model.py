"""
this scipt contains two battery models
an Equivalent circuit model
an electro-thermal-aging model
"""
import math
import pickle
from scipy.interpolate import interp1d


class Battery:
    def __init__(self):
        self.timestep = 1
        self.cell_n = 2       # n串, 6串对应最大输出功率是150kW
        self.maxpower = self.cell_n * 25 * 1000  # W      50kW, 2.473kWh
        # used in electric model
        self.r0 = 0.0031  # Rs
        self.r1 = 0.0062
        self.r2 = 0.0054
        self.c1 = 8710.0
        self.c2 = 258.4211
        # used in heating model
        self.Cn = 2.422     # Ah
        self.Cc = 62.7
        self.Cs = 41
        self.Ru = 5.095
        self.Rc = 1.94
        # used in ageing model
        Ic_rate = [0.5, 2, 6, 10]
        Bc_data = [31630, 21681, 12934, 15512]
        self.Bc_func = interp1d(Ic_rate, Bc_data, kind='linear', fill_value='extrapolate')
        data_dir = "./common/data/"
        self.ocv_func = pickle.load(open(data_dir+'ocv.pkl', 'rb'))

    def run_cell(self, P_batt, paras_list):
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a, Voc, V1, V2]
        SOC = paras_list[0]
        SOH = paras_list[1]
        Tep_c = paras_list[2]
        Tep_s = paras_list[3]
        Tep_a = paras_list[4]
        Voc = paras_list[5]     # initial SOC 0.6: 3.237 V
        V1 = paras_list[6]
        V2 = paras_list[7]
        # battery power limit   # P_batt is in W
        if P_batt > self.maxpower:
            P_batt = self.maxpower
        if P_batt < -self.maxpower:
            P_batt = -self.maxpower
        # battery pack of 168*6 cells
        cell_num = 168 * self.cell_n
        P_cell = P_batt/cell_num  # in W
        # print('cell power: %.4f'%P_cell)
        V_3 = Voc+V1+V2
        delta = V_3**2-4*self.r0*P_cell
        if delta < 0:
            I_batt = V_3/(2*self.r0)
        else:
            I_batt = (V_3-math.sqrt(delta))/(2*self.r0)     # P>0 -> I>0 -> dsoc < 0
        Ic_rate = abs(I_batt/self.Cn)
        cell_heat = I_batt*(V1+V2+self.r0*I_batt)  # H(t)
        soc_deriv = self.timestep*(I_batt/3600/self.Cn)
        v1_deriv = self.timestep*(-V1/self.r1/self.c1+I_batt/self.c1)
        v2_deriv = self.timestep*(-V2/self.r2/self.c2+I_batt/self.c2)
        tc_deriv = self.timestep*(((Tep_s-Tep_c)/self.Rc+cell_heat)/self.Cc)
        ts_deriv = self.timestep*(((Tep_c-Tep_s)/self.Rc+(Tep_a-Tep_s)/self.Ru)/self.Cs)
        # electric model
        SOC_new = SOC-soc_deriv
        if SOC_new >= 1:
            Voc_new = self.ocv_func(1.0)  # for a cell#
            fail = True
            # SOC_new = 1.0
        elif SOC_new <= 0.01:
            Voc_new = self.ocv_func(0.01)
            fail = True
            # SOC_new = 0.001
        else:
            Voc_new = self.ocv_func(SOC_new)  # for a cell
            fail = False
        # print('SOC: %.6f'%SOC_new)
        Voc_new = Voc_new*13.87/168
        V1_new = V1+v1_deriv
        V2_new = V2+v2_deriv
        # terminal voltage
        Vt_new = Voc_new+V1_new+V2_new+self.r0*I_batt
        power_out = Vt_new*I_batt
        # thermal model
        Tep_c_new = Tep_c+tc_deriv
        Tep_s_new = Tep_s+ts_deriv
        Tep_a_new = (Tep_c_new+Tep_s_new)/2
        if Tep_a_new > 60:
            Tep_a_new = 60
        # aging model
        Bc = self.Bc_func(Ic_rate)
        E = 31700-370.3*Ic_rate
        T = Tep_a_new+273.15
        Ah = (20/Bc/math.exp(-E/8.31/T))**(1/0.55)  # z = 0.55, ideal_gas_constant = 8.31
        N1 = 3600*Ah/self.Cn
        dsoh = self.timestep*(abs(I_batt/2/N1/self.Cn))
        SOH_new = SOH-dsoh
    
        out_info = {'SOC': SOC_new, 'SOH': SOH_new,
                    'cell_OCV': Voc_new, 'cell_Vt': Vt_new, 'cell_V_3': V_3,
                    'cell_V1': V1_new, 'cell_V2': V2_new,
                    'I': I_batt, 'I_c': Ic_rate, 'cell_power_out': power_out,
                    'P_batt': P_batt/1000, 'tep_a': Tep_a_new, 'dsoh': dsoh}
        paras_list_new = [SOC_new, SOH_new, Tep_c_new, Tep_s_new, Tep_a_new, Voc_new, V1_new, V2_new]
        done = fail
        return paras_list_new, dsoh, I_batt, done, out_info
    