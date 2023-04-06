#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data from Excel file
FRU = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Management Ranking calculation.xlsx", sheet_name=0)
PU_QP = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Management Ranking calculation.xlsx", sheet_name=1)
FPPP = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Management Ranking calculation.xlsx", sheet_name=2)
GMS = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Management Ranking calculation.xlsx", sheet_name=3)
PCS = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Management Ranking calculation.xlsx", sheet_name=4)

# Split the data into input and output
y = FRU[['FRU']]
x = FRU[['Capex_avg', 'Opex_avg']]
y11 = PU_QP[['PU']]
x11 = PU_QP[['Publications', 'faculty_2018']]
y12 = PU_QP[['QP']]
x12 = PU_QP[['Publications', 'Citations', 'Top25','faculty_2018']]
y2 = FPPP[['FPPP']]
x2 = FPPP[['Research', 'Consultancy', 'Executive']]
y3 = GMS[['GMS']]
x3 = GMS[['Salary', 'Placed']]
y4 = PCS[['PCS']]
x4 = PCS[['A', 'B', 'C']]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x11_train, x11_test, y11_train, y11_test = train_test_split(x11,y11, test_size=0.2,random_state=31)
x12_train, x12_test, y12_train, y12_test = train_test_split(x12,y12, test_size=0.2,random_state=31)
x2_train, x2_test, y2_train, y2_test=train_test_split(x2,y2, test_size=0.2,random_state=31)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size=0.2,random_state=31)
x4_train, x4_test, y4_train, y4_test=train_test_split(x4,y4, test_size=0.2,random_state=31)

# Train the random forest regression model for prediction
fru_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fru_rf = fru_model.fit(x_train, y_train.values.ravel())
pu_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pu_rf = pu_model.fit(x11_train, y11_train.values.ravel())
qp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
qp_rf = qp_model.fit(x12_train, y12_train.values.ravel())
fppp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fppp_rf = fppp_model.fit(x2_train, y2_train.values.ravel())
gms_model = RandomForestRegressor(n_estimators=1000, random_state=31)
gms_rf = gms_model.fit(x3_train, y3_train.values.ravel())
pcs_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pcs_rf = pcs_model.fit(x4_train, y4_train.values.ravel())

def predict_SS(SI, TE, ft):
    students = max(SI,TE)
    Ratio = min(TE/SI,1)
    if students >=720:
        criteria = 15
    elif students <720 and students >=480:
        criteria = 12
    elif students <480 and students >=240:
        criteria = 9
    else:
        criteria = 6
    student_score = round(Ratio*criteria,2)
    if ft>=150:
        PhD_score = 5
    elif ft<150 and ft>=100:
        PhD_score = 4
    elif ft<100 and ft>=50:
        PhD_score = 3
    elif ft<50 and ft>=20:
        PhD_score = 2
    elif ft<20 and ft>=10:
        PhD_score = 1
    elif ft<10 and ft>=5:
        PhD_score = 0.5
    else:
        PhD_score = 0
    SS = student_score + PhD_score
    return SS

def predict_FSR(Nfaculty, SI, ft):
    N = SI + ft
    ratio = Nfaculty / N
    FSR = min((round(30 * (15 * ratio),2)),30)
    #if FSR > 30:
        #FSR = 30
    #else:
        #FSR = FSR
    if ratio < 0.02:
        FSR = 0
    return FSR

def predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3):
    calc_FSR=SI/15
    Faculty = max(calc_FSR,Nfaculty)
    FRA=(phd/Faculty)
    if FRA<0.95:
        FQ=10*(FRA/0.95)
    else:
        FQ=10
    F1=ft_exp1/Faculty
    F2=ft_exp2/Faculty
    F3=ft_exp3/Faculty
    FE_cal=(3*min((3*F1),1))+(3*min((3*F2),1))+(4*min((3*F3),1))
    if F1==F2==F3:
        FE=10
    else:
        FE = FE_cal
    FQE=round(FQ+FE,2)
    return FQE

def predict_FRU(Capex_avg, Opex_avg):
    FRU = round(fru_model.predict([[Capex_avg, Opex_avg]])[0],2)
    return FRU

def predict_PU(Publications, faculty_2018):
    PU = round(pu_model.predict([[Publications, faculty_2018]])[0],2)
    return PU

def predict_QP(Publications,Citations,Top25,faculty_2018):
    QP = round(qp_model.predict([[Publications,Citations,Top25,faculty_2018]])[0],2)
    return QP

def predict_FPPP(Research,Consultancy,Executive):
    FPPP = round(fppp_model.predict([[Research,Consultancy,Executive]])[0],2)
    return FPPP

def predict_GPH(Placement,HS,UG_sum):
    GPH = round((40*(Placement/UG_sum)+(HS/UG_sum)),2)
    return GPH

def predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3):
    year1 = graduated1/si1
    year2 = graduated2/si2
    year3 = graduated3/si3
    avg=(year1+year2+year3)/3
    a=avg/0.8
    GUE=round((min(a,1)*20),2)
    return GUE

def predict_GMS(Salary,Placed):
    GMS = round(gms_model.predict([[Salary,Placed]])[0],2)
    return GMS

def predict_RD(SI,TE,other_state):
    students = max(SI,TE)
    RD = round((other_state/students)*30, 2)
    return RD

def predict_WD(WS,WF,SI,TE,Nfaculty):
    calc_FSR=(max(SI,TE))/15
    Faculty = max(calc_FSR,Nfaculty)
    student_ratio=WS/SI
    faculty_ratio=WF/Faculty
    a1=min(((student_ratio)/0.5),1)
    b1=min(((faculty_ratio)/0.2),1)
    WD=round(((15*a1)+(15*b1)),2)
    return WD

def predict_ESCS(SI, socio_economic, reimbursed):
    Reimbursed_ratio = (reimbursed/socio_economic)
    Student_ratio = (socio_economic/SI)
    ESCS = round((Reimbursed_ratio*Student_ratio*10),2)
    #ESCS = tuple(round(val, 2) for val in (Reimbursed_ratio, Student_ratio*10))
    return ESCS

def predict_PCS(A,B,C):
    PCS = round(pcs_model.predict([[A,B,C]])[0],2)
    return PCS

def predict_all(y11,y12,y21,y22,y23,y24,y25, TE1,TE2,ft,Nfaculty, phd,ft_exp1,ft_exp2,ft_exp3, L1,L2,L3,Lab1,Lab2,Lab3,O1,O2,O3, S1,S2,S3,I1,I2,I3,Seminar1,Seminar2,Seminar3,faculty_2018,Publications, Citations,Top25, RF1,RF2,RF3,CF1,CF2,CF3,Executive1,Executive2,Executive3, Placement11, Placement12, Placement13,Placement21,Placement22,Placement23, HS11, HS12, HS13,HS21,HS22,HS23,si11,si12,si13,si21,si22,si23, graduated11, graduated12, graduated13, graduated21, graduated22, graduated23, Salary1, Salary2, Salary3, state1,state2, WS,WF, Socially_challenged1,Socially_challenged2,economically_challenged1,economically_challenged2,reimbursed1,reimbursed2, A,B,C, pr):
    SI = y11+y12+y21+y22+y23+y24+y25
    TE = TE1+TE2
    ss = predict_SS(SI, TE, ft)
    fsr = predict_FSR(Nfaculty, SI, ft)
    fqe = predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3)
    cap_year1 = (L1+Lab1+O1)/SI
    cap_year2 = (L2+Lab2+O2)/SI
    cap_year3 = (L3+Lab3+O3)/SI
    Capex_avg = ((cap_year1+cap_year2+cap_year3)/3)
    opr_year1 = (S1+I1+Seminar1)/SI
    opr_year2 = (S2+I2+Seminar2)/SI
    opr_year3 = (S3+I3+Seminar3)/SI
    Opex_avg = ((opr_year1 + opr_year2 + opr_year3)/3)
    fru = predict_FRU(Capex_avg, Opex_avg)
    tlr = round(ss + fsr + fqe + fru, 2)
    pu = predict_PU(Publications, faculty_2018)
    qp = predict_QP(Publications,Citations,Top25,faculty_2018)
    Research = (((RF1/Nfaculty)+(RF2/Nfaculty)+(RF3/Nfaculty))/3)
    Consultancy = (((CF1/Nfaculty)+(CF2/Nfaculty)+(CF3/Nfaculty))/3)
    Executive = (((Executive1/Nfaculty)+(Executive2/Nfaculty)+(Executive3/Nfaculty))/3)
    fppp = predict_FPPP(Research,Consultancy,Executive)
    rp = round((pu + qp + fppp),2)
    Placement = (Placement11 + Placement12 + Placement13 + Placement21 + Placement22 + Placement23)
    HS = (HS11 + HS12 + HS13 + HS21 + HS22 + HS23)
    UG_sum = si11+si12+si13+si21+si22+si23
    gph = predict_GPH(Placement,HS,UG_sum)
    graduated1 = (graduated11+graduated21)
    graduated2 = (graduated12+graduated22)
    graduated3 = (graduated13+graduated23)
    si1 = si11+si21
    si2 = si12+si22
    si3 = si13+si23
    gue = predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3)
    Salary = Salary1+ Salary2+ Salary3
    Placed = Placement
    gms = predict_GMS(Salary,Placed)
    go = round((gph + gue + gms),2)
    other_state = (state1+ state2)
    rd = predict_RD(SI,TE,other_state)
    wd = predict_WD(WS,WF,SI,TE,Nfaculty)
    socio_economic = (Socially_challenged1+Socially_challenged2+economically_challenged1+economically_challenged2)
    reimbursed = reimbursed1+reimbursed2
    escs = predict_ESCS(SI, socio_economic, reimbursed)
    pcs = predict_PCS(A,B,C)
    oi = round((rd + wd + escs + pcs),2)
    Overall_score = round(((tlr * 0.3) + (rp * 0.3) + (go * 0.2) + (oi * 0.1) + (pr * 0.1)),2)
    
    return(ss,fsr,fqe,fru,tlr,pu,qp,fppp,rp,gph,gue,gms,go,rd,wd,escs,pcs,oi,Overall_score)

# Create a Gradio interface
inputs = [
    gr.components.Number(label="Sanctioned Intake PG 2 year-year1", value=542),
    gr.components.Number(label="Sanctioned Intake PG 2 year-year2", value=541),
    gr.components.Number(label="Sanctioned Intake PG integrated-year1", value=0),
    gr.components.Number(label="Sanctioned Intake PG integrated-year2", value=0),
    gr.components.Number(label="Sanctioned Intake PG integrated-year3", value=0),
    gr.components.Number(label="Sanctioned Intake PG integrated-year4", value=0),
    gr.components.Number(label="Sanctioned Intake PG integrated-year5", value=0),
    gr.components.Number(label="Total Enrollment - PG 2 year", value=1081),
    gr.components.Number(label="Total Enrollment - PG integrated", value=0),
    gr.components.Number(label="Number of PhD Enrolled Full Time", value=97),
    gr.components.Number(label="No. of Full Time Regular Faculty", value=87),
    gr.components.Number(label="No. of faculty with PhD", value=80),
    gr.components.Number(label="No. of full time regular faculty with Experience up to 8 years", value=27),
    gr.components.Number(label="No. of full time regular faculty with Experience between 8+ to 15 years", value=30),
    gr.components.Number(label="No. of full time regular faculty with Experience > 15 years", value=30),
    gr.components.Number(label="Annual Expenditure on Library-year1", value=58100000),
    gr.components.Number(label="Annual Expenditure on Library-year2", value=38000000),
    gr.components.Number(label="Annual Expenditure on Library-year3", value=33600000),
    gr.components.Number(label="Annual Expenditure on Laboratory-year1", value=20568179),
    gr.components.Number(label="Annual Expenditure on Laboratory-year2", value=4417597),
    gr.components.Number(label="Annual Expenditure on Laboratory-year3", value=35556500),
    gr.components.Number(label="Annual Expenditure on Others-year1", value=12263691),
    gr.components.Number(label="Annual Expenditure on Others-year2", value=18700000),
    gr.components.Number(label="Annual Expenditure on Others-year3", value=42200000),
    gr.components.Number(label="Annual Expenditure on Salary-year1", value=339400000),
    gr.components.Number(label="Annual Expenditure on Salary-year2", value=343000000),
    gr.components.Number(label="Annual Expenditure on Salary-year3", value=336600000),
    gr.components.Number(label="Annual Expenditure on Infrastructure-year1", value=885300000),
    gr.components.Number(label="Annual Expenditure on Infrastructure-year2", value=940100000),
    gr.components.Number(label="Annual Expenditure on Infrastructure-year3", value=1395100000),
    gr.components.Number(label="Annual Expenditure on Seminar-year1", value=23000000),
    gr.components.Number(label="Annual Expenditure on Seminar-year2", value=27800000),
    gr.components.Number(label="Annual Expenditure on Seminar-year3", value=29700000),
    gr.components.Number(label="No. of Full Time Regular Faculty", value=87),
    gr.components.Number(label="No. of Publications", value=60),
    gr.components.Number(label="No. of citations", value=60),
    gr.components.Number(label="No. of Top25 percentage", value=60),
    gr.components.Number(label="Amount received in sponsored research - year1", value=29557741),
    gr.components.Number(label="Amount received in sponsored research - year2", value=55891434),
    gr.components.Number(label="Amount received in sponsored research - year3", value=112200598),
    gr.components.Number(label="Amount received in consultancy projects - year1", value=25325000),
    gr.components.Number(label="Amount received in consultancy projects - year2", value=116740000),
    gr.components.Number(label="Amount received in consultancy projects - year3", value=189056000),
    gr.components.Number(label="Amount earned in EDP - year1", value=584100000),
    gr.components.Number(label="Amount earned in EDP - year2", value=597984756),
    gr.components.Number(label="Amount earned in EDP - year3", value=419501486),
    gr.components.Number(label="No. of students placed (PG-2yr) - year1", value=505),
    gr.components.Number(label="No. of students placed (PG-2yr) - year2", value=499),
    gr.components.Number(label="No. of students placed (PG-2yr) - year3", value=523),
    gr.components.Number(label="No. of students placed (PG integrated) - year1", value=505),
    gr.components.Number(label="No. of students placed (PG integrated) - year2", value=499),
    gr.components.Number(label="No. of students placed (PG integrated) - year3", value=523),
    gr.components.Number(label="No. of students selected for HS (PG-2yr) - year1", value=0),
    gr.components.Number(label="No. of students selected for HS (PG-2yr) - year2", value=0),
    gr.components.Number(label="No. of students selected for HS (PG-2yr) - year3", value=0),
    gr.components.Number(label="No. of students selected for HS (PG integrated) - year1", value=0),
    gr.components.Number(label="No. of students selected for HS (PG integrated) - year2", value=0),
    gr.components.Number(label="No. of students selected for HS (PG integrated) - year3", value=0),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year3", default=60),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year1", value=505),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year2", value=499),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year3", value=523),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year1", value=0),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year2", value=0),
    gr.components.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year3", value=0),
    gr.components.Number(label="Total Median Salary - year1", value=2200000),
    gr.components.Number(label="Total Median Salary - year2", value=2501000),
    gr.components.Number(label="Total Median Salary - year3", value=2767000),
    gr.components.Number(label="Total Enrollment of students from other states - PG 2 year", value=992),
    gr.components.Number(label="Total Enrollment of students from other states - PG integrated", value=0),
    gr.components.Number(label="No. of women students", value=326),
    gr.components.Number(label="No. of women faculty", value=30),
    gr.components.Number(label="No. of students socially challenged - PG 2 year", value=536),
    gr.components.Number(label="No. of students socially challenged - PG integrated", value=0),
    gr.components.Number(label="No. of students economically challenged - PG 2 year", value=32),
    gr.components.Number(label="No. of students economically challenged - PG integrated", value=0),
    gr.components.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG 2 year", value=53),
    gr.components.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG integrated", value=0),
    gr.components.Number(label="Lifts/Ramps", value=80),
    gr.inputs.Dropdown(choices = ["1", "0"],label="Walking aids", default=1),
    gr.components.Number(label="Specially designed toilets for handicapped students", value=80),
    gr.components.Number(label="PR", value=94.74)
]
output1 = gr.components.Textbox(label="SS")
output2 = gr.components.Textbox(label="FSR")
output3 = gr.components.Textbox(label="FQE")
output4 = gr.components.Textbox(label="FRU")
output_tlr = gr.components.Textbox(label="TLR")
output5 = gr.components.Textbox(label="PU")
output6 = gr.components.Textbox(label="QP")
output7 = gr.components.Textbox(label="FPPP")
output_rp = gr.components.Textbox(label="RP")
output8 = gr.components.Textbox(label="GPH")
output9 = gr.components.Textbox(label="GUE")
output10 = gr.components.Textbox(label="GMS")
output_go = gr.components.Textbox(label="GO")
output11 = gr.components.Textbox(label="RD")
output12 = gr.components.Textbox(label="WD")
output13 = gr.components.Textbox(label="ESCS")
output14 = gr.components.Textbox(label="PCS")
output_oi = gr.components.Textbox(label="OI")
output_score = gr.components.Textbox(label="Overall_score")
gradio_interface = gr.Interface(fn=predict_all, inputs=inputs, outputs=[output1, output2, output3, output4, output_tlr, output5, output6, output7, output_rp, output8, output9, output10, output_go, output11, output12, output13, output14, output_oi, output_score], title="Management NIRF Score Calculation", 
                                description="Enter the input parameters to predict Overall score")

gradio_interface.launch(share=True)


# In[ ]:




