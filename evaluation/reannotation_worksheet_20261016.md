# Gold re-annotation worksheet (corpus snapshot 2026-10-16)

6 items re-pointed automatically (verbatim snippet, single chunk). 30 need your choice.

For each item, pick the ONE candidate that best answers the question as the old gold did, write its letter in `evaluation/reannotation_answers_20261016.json`. Use `"X"` if no candidate answers it (the query will be flagged for a rewrite). Judge from the label text only.

When the question names a generic shared by two products (natalizumab: Tysabri and Tyruko), prefer the chunk from the drug in `expected_drug`. Where a real boxed-warning chunk now exists and it is what the question asks about, prefer it. Use the `note` field for anything unusual.


---
## AE-006  (Mayzent · expected section: adverse_reactions)

**Question:** Are there cardiac side effects with siponimod?

**Old gold text** (35c8afc335d3, adverse_reactions):
> 6 ADVERSE REACTIONS The following serious adverse reactions are described elsewhere in the labeling: Infections [see Warnings and Precautions (5.1)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.2)] Macular Edema [see Warnings and Precautions (5.3)] Bradyarrhythmia and Atrioventricular Conduction Delays [see Warnings and Precautions (5.4)] Respiratory Effects [see Warnings and Precautions (5.5)] Liver Injury [see Warnings and Precautions (5.6)] Cutaneous Malignancie

**A** `35c8afc335d3` · Mayzent · adverse_reactions · chunk 1/5
> 6 ADVERSE REACTIONS The following serious adverse reactions are described elsewhere in the labeling: Infections [see Warnings and Precautions (5.1)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.2)] Macular Edema [see Warnings and Precautions (5.3)] Bradyarrhythmia and Atrioventricular Conduction Delays [see Warnings and Precautions (5.4)] Respiratory Effects [see Warnings and Precautions (5.5)] Liver Injury [see Warnings and Precautions (5.6)] Cutaneous Malignancie

**B** `63330f818773` · Mayzent · adverse_reactions · chunk 5/5
> edema, rash, and urticaria [see Contraindications (4) and Warnings and Precautions (5.8)]. Infections and Infestations: Progressive multifocal leukoencephalopathy [see Warnings and Precautions (5.2)].

**C** `311b374ff68b` · Mayzent · warnings_and_precautions · chunk 1/16
> 5 WARNINGS AND PRECAUTIONS Infections: MAYZENT may increase the risk. Obtain a complete blood count (CBC) before initiating treatment. Monitor for infection during treatment. Do not start in patients with active infection. (5.1) Progressive Multifocal Leukoencephalopathy (PML): Withhold MAYZENT at the first sign or symptom of PML. (5.2) Macular Edema: Increases the risk of macular edema. Obtain a baseline evaluation of the fundus, including the macula, near the start of treatment with MAYZENT. C


---
## WP-005  (Tysabri · expected section: warnings_and_precautions)

**Question:** What warnings exist for PML with natalizumab?

**Old gold text** (af30f0f2afbb, warnings_and_precautions):
> 5 WARNINGS AND PRECAUTIONS •Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYRUKO if these infections occur and treat appropriately. (5.3) •Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYRUKO in patients with evidence of liver injury. (5.4) •Hypersensitivity reactions: Serious 

**A** `03b4ce4527bd` · Tyruko · warnings_and_precautions · chunk 1/14
> 5 WARNINGS AND PRECAUTIONS •Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYRUKO if these infections occur and treat appropriately. (5.3) •Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYRUKO in patients with evidence of liver injury. (5.4) •Hypersensitivity reactions: Serious 

**B** `af30f0f2afbb` · Tysabri · warnings_and_precautions · chunk 1/14
> 5 WARNINGS AND PRECAUTIONS Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYSABRI if these infections occur and treat appropriately (5.3) Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYSABRI in patients with evidence of liver injury (5.4) Hypersensitivity reactions: Serious hyp

**C** `b7c71cf92203` · Tyruko · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY Natalizumab products increase the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with T


---
## SP-003  (Tysabri · expected section: use_in_specific_populations)

**Question:** What are the pregnancy warnings for natalizumab?

**Old gold text** (977285fd2098, use_in_specific_populations):
> 8 USE IN SPECIFIC POPULATIONS Pregnancy: Can cause fetal harm. (5.8, 8.1) *Biosimilar means that the biological product is approved based on data demonstrating that it is highly similar to an FDA-approved biological product, known as a reference product, and that there are no clinically meaningful differences between the biosimilar product and the reference product. Biosimilarity of TYRUKO has been demonstrated for the condition(s) of use (e.g., indication(s), dosing regimen(s)), strength(s), do

**A** `36f56553f202` · Tyruko · use_in_specific_populations · chunk 1/4
> 8 USE IN SPECIFIC POPULATIONS Pregnancy: Can cause fetal harm. (5.8, 8.1) *Biosimilar means that the biological product is approved based on data demonstrating that it is highly similar to an FDA-approved biological product, known as a reference product, and that there are no clinically meaningful differences between the biosimilar product and the reference product. Biosimilarity of TYRUKO has been demonstrated for the condition(s) of use (e.g., indication(s), dosing regimen(s)), strength(s), do

**B** `977285fd2098` · Tysabri · use_in_specific_populations · chunk 1/4
> 8 USE IN SPECIFIC POPULATIONS Pregnancy: Can cause fetal harm. (5.8, 8.1) 8.1 Pregnancy Risk Summary There are no adequate data on the risk of major birth defects, miscarriage, or other adverse maternal outcomes associated with the use of TYSABRI in pregnant women. Adverse fetal outcomes of neonatal thrombocytopenia and anemia have been reported (see Clinical Considerations). In animal studies, administration of natalizumab during pregnancy produced fetal immunologic and hematologic effects in m

**C** `71a9166425dc` · Tysabri · boxed_warning · chunk 3/3
> the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)


---
## SP-006  (Gabapentin · expected section: use_in_specific_populations)

**Question:** What about using gabapentin in renal impairment?

**Old gold text** (2c0a0144df12, use_in_specific_populations):
> cretion, the larger treatment effect observed in patients ≥75 years may be a consequence of increased gabapentin exposure for a given dose that results from an age-related decrease in renal function. However, other factors cannot be excluded. The types and incidence of adverse reactions were similar across age groups except for peripheral edema and ataxia, which tended to increase in incidence with age. Clinical studies of gabapentin in epilepsy did not include sufficient numbers of subjects age

**A** `2c0a0144df12` · Gabapentin · use_in_specific_populations · chunk 5/6
> cretion, the larger treatment effect observed in patients ≥75 years may be a consequence of increased gabapentin exposure for a given dose that results from an age-related decrease in renal function. However, other factors cannot be excluded. The types and incidence of adverse reactions were similar across age groups except for peripheral edema and ataxia, which tended to increase in incidence with age. Clinical studies of gabapentin in epilepsy did not include sufficient numbers of subjects age

**B** `f7097d747cb8` · Gabapentin · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS Gabapentin is contraindicated in patients who have demonstrated hypersensitivity to the drug or its ingredients. Known hypersensitivity to gabapentin or its ingredients ( 4)

**C** `2d46745647fc` · Gabapentin · use_in_specific_populations · chunk 6/6
> tric patients with renal insufficiency have not been studied. Dosage adjustment in patients undergoing hemodialysis is necessary [see Dosage and Administration (2.3)and Clinical Pharmacology (12.3)] .


---
## PT-006  (Gilenya · expected section: warnings_and_precautions)

**Question:** Will fingolimod affect my heart rate?

**Old gold text** (65ec8e58d5e1, warnings_and_precautions):
> apse occurs during treatment or after discontinuation. Obtain imaging and begin treatment as needed. (5.10) Increased Blood Pressure (BP): Monitor BP in adult and pediatric patients during treatment. (5.11) Malignancies: Skin examination prior to or shortly after the start of treatment and periodically thereafter is recommended. Suspicious skin lesions should be evaluated. (5.12) 5.1 Bradyarrhythmia and Atrioventricular Blocks Because of a risk for bradyarrhythmia and AV blocks, patients should 

**A** `65ec8e58d5e1` · Gilenya · warnings_and_precautions · chunk 2/17
> apse occurs during treatment or after discontinuation. Obtain imaging and begin treatment as needed. (5.10) Increased Blood Pressure (BP): Monitor BP in adult and pediatric patients during treatment. (5.11) Malignancies: Skin examination prior to or shortly after the start of treatment and periodically thereafter is recommended. Suspicious skin lesions should be evaluated. (5.12) 5.1 Bradyarrhythmia and Atrioventricular Blocks Because of a risk for bradyarrhythmia and AV blocks, patients should 

**B** `e89057faa937` · Gilenya · adverse_reactions · chunk 1/6
> 6 ADVERSE REACTIONS The following serious adverse reactions are described elsewhere in labeling: Bradyarrhythmia and Atrioventricular Blocks [see Warnings and Precautions (5.1)] Infections [see Warnings and Precautions (5.2)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.3)] Macular Edema [see Warnings and Precautions (5.4)] Liver Injury [see Warnings and Precautions (5.5)] Posterior Reversible Encephalopathy Syndrome [see Warnings and Precautions (5.6)] Respiratory

**C** `c85cc8670fc7` · Gilenya · drug_interactions · chunk 2/3
> ncluding corticosteroids) are expected to increase the risk of immunosuppression, and the risk of additive immune system effects must be considered if these therapies are coadministered with GILENYA. When switching from drugs with prolonged immune effects, such as natalizumab, teriflunomide or mitoxantrone, the duration and mode of action of these drugs must be considered to avoid unintended additive immunosuppressive effects when initiating GILENYA [see Warnings and Precautions (5.2)]. 7.5 Drug


---
## CI-001  (Tysabri · expected section: contraindications)

**Question:** What are the contraindications for natalizumab?

**Old gold text** (6f6ee168435d, contraindications):
> 4 CONTRAINDICATIONS •TYRUKO is contraindicated in patients who have or have had progressive multifocal leukoencephalopathy (PML) [see Warnings and Precautions (5.1)]. •TYRUKO is contraindicated in patients who have had a hypersensitivity reaction to natalizumab products or any of the ingredients in TYRUKO. Observed reactions range from urticaria to anaphylaxis [see Warnings and Precautions (5.5)]. •Patients who have or have had PML (4) •Patients who have had a hypersensitivity reaction to natali

**A** `01467d6585fe` · Tyruko · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS •TYRUKO is contraindicated in patients who have or have had progressive multifocal leukoencephalopathy (PML) [see Warnings and Precautions (5.1)]. •TYRUKO is contraindicated in patients who have had a hypersensitivity reaction to natalizumab products or any of the ingredients in TYRUKO. Observed reactions range from urticaria to anaphylaxis [see Warnings and Precautions (5.5)]. •Patients who have or have had PML (4) •Patients who have had a hypersensitivity reaction to natali

**B** `6f6ee168435d` · Tysabri · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS TYSABRI is contraindicated in patients who have or have had progressive multifocal leukoencephalopathy (PML) [see Warnings and Precautions (5.1)]. TYSABRI is contraindicated in patients who have had a hypersensitivity reaction to TYSABRI. Observed reactions range from urticaria to anaphylaxis [see Warnings and Precautions (5.5)]. Patients who have or have had PML (4) Patients who have had a hypersensitivity reaction to TYSABRI (4, 5.3)

**C** `f5ead356b4a4` · Tyruko · warnings_and_precautions · chunk 14/14
> on the effects of vaccination in patients receiving natalizumab products. No data are available on the secondary transmission of infection by live vaccines in patients receiving natalizumab products.

**D** `59ed4675b125` · Tysabri · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY TYSABRI increases the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with TYSABRI [ see

**E** `1fd0911a16fb` · Tysabri · boxed_warning · chunk 2/3
> r patients, and withhold TYSABRI immediately at the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**F** `71a9166425dc` · Tysabri · boxed_warning · chunk 3/3
> the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**G** `b7c71cf92203` · Tyruko · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY Natalizumab products increase the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with T

**H** `6c082b50694f` · Tyruko · boxed_warning · chunk 2/3
> he context of expected benefit when initiating and continuing treatment with TYRUKO (5.1). • Monitor patients, and withhold TYRUKO immediately at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).

**I** `826d818c05d3` · Tyruko · boxed_warning · chunk 3/3
> ely at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).


---
## MD-001  (Tysabri · expected section: warnings_and_precautions)

**Question:** Between natalizumab and ocrelizumab, which carries a PML risk?

**Old gold text** (445299813365, warnings_and_precautions):
> am. In multiple sclerosis patients, an MRI scan should be obtained prior to initiating therapy with TYRUKO. This MRI may be helpful in differentiating subsequent multiple sclerosis symptoms from PML. In Crohn’s disease patients, a baseline brain MRI may also be helpful to distinguish pre-existent lesions from newly developed lesions, but brain lesions at baseline that could cause diagnostic difficulty while on natalizumab product therapy are uncommon. For diagnosis of PML, an evaluation includin

**A** `250056527e22` · Tyruko · warnings_and_precautions · chunk 5/14
> am. In multiple sclerosis patients, an MRI scan should be obtained prior to initiating therapy with TYRUKO. This MRI may be helpful in differentiating subsequent multiple sclerosis symptoms from PML. In Crohn’s disease patients, a baseline brain MRI may also be helpful to distinguish pre-existent lesions from newly developed lesions, but brain lesions at baseline that could cause diagnostic difficulty while on natalizumab product therapy are uncommon. For diagnosis of PML, an evaluation includin

**B** `445299813365` · Tysabri · warnings_and_precautions · chunk 5/14
> m. In multiple sclerosis patients, an MRI scan should be obtained prior to initiating therapy with TYSABRI. This MRI may be helpful in differentiating subsequent multiple sclerosis symptoms from PML. In Crohn's disease patients, a baseline brain MRI may also be helpful to distinguish pre-existent lesions from newly developed lesions, but brain lesions at baseline that could cause diagnostic difficulty while on TYSABRI therapy are uncommon. For diagnosis of PML, an evaluation including a gadolini

**C** `f5ead356b4a4` · Tyruko · warnings_and_precautions · chunk 14/14
> on the effects of vaccination in patients receiving natalizumab products. No data are available on the secondary transmission of infection by live vaccines in patients receiving natalizumab products.

**D** `59ed4675b125` · Tysabri · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY TYSABRI increases the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with TYSABRI [ see

**E** `1fd0911a16fb` · Tysabri · boxed_warning · chunk 2/3
> r patients, and withhold TYSABRI immediately at the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**F** `71a9166425dc` · Tysabri · boxed_warning · chunk 3/3
> the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**G** `b7c71cf92203` · Tyruko · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY Natalizumab products increase the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with T

**H** `6c082b50694f` · Tyruko · boxed_warning · chunk 2/3
> he context of expected benefit when initiating and continuing treatment with TYRUKO (5.1). • Monitor patients, and withhold TYRUKO immediately at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).

**I** `826d818c05d3` · Tyruko · boxed_warning · chunk 3/3
> ely at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).


---
## CI-003  (Gilenya · expected section: contraindications)

**Question:** Who should not take fingolimod?

**Old gold text** (758f661c7931, contraindications):
> 4 CONTRAINDICATIONS Fingolimod is contraindicated in patients who have: in the last 6 months experienced myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization or Class III/IV heart failure a history or presence of Mobitz Type II second-degree or third-degree AV block or sick sinus syndrome, unless patient has a functioning pacemaker [see Warnings and Precautions (5.1)] a baseline QTc interval ≥500 msec cardiac arrhy

**A** `758f661c7931` · Gilenya · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS GILENYA is contraindicated in patients who have: in the last 6 months experienced myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization or Class III/IV heart failure a history or presence of Mobitz Type II second-degree or third-degree AV block or sick sinus syndrome, unless patient has a functioning pacemaker [see Warnings and Precautions (5.1)] a baseline QTc interval ≥ 500 msec cardiac arrhyth

**B** `e8eebaaff03e` · Gilenya · adverse_reactions · chunk 6/6
> uding mycosis fungoides), Kaposi’s sarcoma, squamous cell carcinoma [see Warnings and Precautions (5.12)] Skin and Subcutaneous Tissue Disorders: Hypersensitivity [see Warnings and Precautions (5.14)]

**C** `25e22f2e4fb5` · Gilenya · adverse_reactions · chunk 5/6
> pecified (including cysts and polyps): Melanoma, Merkel cell carcinoma, cutaneous T-cell lymphoma (including mycosis fungoides), Kaposi’s sarcoma, squamous cell carcinoma [see Warnings and Precautions (5.12)] Skin and Subcutaneous Tissue Disorders: Hypersensitivity [see Warnings and Precautions (5.14)]


---
## CI-005  (Zeposia · expected section: contraindications)

**Question:** What are the contraindications for ozanimod?

**Old gold text** (64f3839386fa, contraindications):
> 4 CONTRAINDICATIONS ZEPOSIA is contraindicated in patients who: •In the last 6 months, have experienced a myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization, or Class III or IV heart failure [see Warnings and Precautions (5.3) ] •Have the presence of Mobitz type II second-degree or third degree atrioventricular (AV) block, sick sinus syndrome, or sino-atrial block, unless the patient has a functioning pacemaker [

**A** `64f3839386fa` · Zeposia · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS ZEPOSIA is contraindicated in patients who: In the last 6 months, have experienced a myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization, or Class III or IV heart failure [see Warnings and Precautions (5.3) ] Have the presence of Mobitz type II second-degree or third degree atrioventricular (AV) block, sick sinus syndrome, or sino-atrial block, unless the patient has a functioning pacemaker [se

**B** `9e278dfabb0b` · Zeposia · use_in_specific_populations · chunk 5/5
> and its active metabolites were not evaluated in patients with severe hepatic impairment (Child-Pugh class C). Therefore, use of ZEPOSIA in patients with severe hepatic impairment is not recommended.

**C** `9805a2de2126` · Zeposia · dosage_and_administration · chunk 1/4
> 2 DOSAGE AND ADMINISTRATION Assessments are required prior to initiating ZEPOSIA. ( 2.1) Titration is required for treatment initiation. ( 2.2) The recommended maintenance dosage is 0.92 mg orally once daily. ( 2.2) The recommended maintenance dosage in patients with mild or moderate chronic hepatic impairment (Child-Pugh class A or B) is 0.92 mg once every other day. ( 2.3) If a dose is missed within the first 2 weeks of treatment, reinitiate with the titration regimen. If a dose is missed afte


---
## CI-006  (Tecfidera · expected section: contraindications)

**Question:** When should dimethyl fumarate not be prescribed?

**Old gold text** (25c3964c080d, contraindications):
> 4 CONTRAINDICATIONS Dimethyl fumarate delayed-release capsules are contraindicated in patients with known hypersensitivity to dimethyl fumarate or to any of the excipients of dimethyl fumarate delayed-release capsules. Reactions have included anaphylaxis and angioedema [see Warnings and Precautions (5.1)]. Known hypersensitivity to dimethyl fumarate or any of the excipients of dimethyl fumarate delayed-release capsules.

**A** `25c3964c080d` · Tecfidera · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS TECFIDERA is contraindicated in patients with known hypersensitivity to dimethyl fumarate or to any of the excipients of TECFIDERA. Reactions have included anaphylaxis and angioedema [see Warnings and Precautions (5.1)]. Known hypersensitivity to dimethyl fumarate or any of the excipients of TECFIDERA. (4)

**B** `5369168c8838` · Tecfidera · adverse_reactions · chunk 3/3
> ns: Herpes zoster infection and other serious opportunistic infections [see Warnings and Precautions (5.3)] Respiratory, Thoracic, and Mediastinal Disorders: Rhinorrhea Skin and Subcutaneous: Alopecia

**C** `a45f19bb7817` · Tecfidera · adverse_reactions · chunk 1/3
> 6 ADVERSE REACTIONS The following important adverse reactions are described elsewhere in labeling: Anaphylaxis and Angioedema [see Warnings and Precautions (5.1)]. Progressive multifocal leukoencephalopathy [see Warnings and Precautions ( 5.2)]. Herpes Zoster and Other Serious Opportunistic Infections [see Warnings and Precautions (5.3)]. Lymphopenia [see Warnings and Precautions (5.4)]. Liver Injury [see Warnings and Precautions (5.5)]. Flushing [see Warnings and Precautions (5.6)]. Serious Gas


---
## AE-002  (Tysabri · expected section: adverse_reactions)

**Question:** What adverse reactions have been reported with natalizumab?

**Old gold text** (da81c145cabe, adverse_reactions):
> 6 ADVERSE REACTIONS The following serious adverse reactions are described below and elsewhere in the labeling: •Progressive Multifocal Leukoencephalopathy (PML) [see Warnings and Precautions (5.1)] •Herpes Infections [see Warnings and Precautions (5.3)] •Hepatotoxicity [see Warnings and Precautions (5.4)] •Hypersensitivity/Antibody Formation [see Warnings and Precautions (5.5)] •Immunosuppression/Infections [see Warnings and Precautions (5.6)] •Hematological Abnormalities [see Warnings and Preca

**A** `16ba27731e8f` · Tyruko · adverse_reactions · chunk 1/10
> 6 ADVERSE REACTIONS The following serious adverse reactions are described below and elsewhere in the labeling: •Progressive Multifocal Leukoencephalopathy (PML) [see Warnings and Precautions (5.1)] •Herpes Infections [see Warnings and Precautions (5.3)] •Hepatotoxicity [see Warnings and Precautions (5.4)] •Hypersensitivity/Antibody Formation [see Warnings and Precautions (5.5)] •Immunosuppression/Infections [see Warnings and Precautions (5.6)] •Hematological Abnormalities [see Warnings and Preca

**B** `da81c145cabe` · Tysabri · adverse_reactions · chunk 1/10
> 6 ADVERSE REACTIONS The following serious adverse reactions are described below and elsewhere in the labeling: Progressive Multifocal Leukoencephalopathy (PML) [see Warnings and Precautions (5.1)] Herpes Infections [see Warnings and Precautions (5.3)] Hepatotoxicity [see Warnings and Precautions (5.4)] Hypersensitivity/Antibody Formation [see Warnings and Precautions (5.5 )] Immunosuppression/Infections [see Warnings and Precautions (5.6 )] Hematological Abnormalities [see Warnings and Precautio

**C** `01467d6585fe` · Tyruko · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS •TYRUKO is contraindicated in patients who have or have had progressive multifocal leukoencephalopathy (PML) [see Warnings and Precautions (5.1)]. •TYRUKO is contraindicated in patients who have had a hypersensitivity reaction to natalizumab products or any of the ingredients in TYRUKO. Observed reactions range from urticaria to anaphylaxis [see Warnings and Precautions (5.5)]. •Patients who have or have had PML (4) •Patients who have had a hypersensitivity reaction to natali


---
## AE-003  (Ocrevus · expected section: adverse_reactions)

**Question:** Does ocrelizumab cause infusion reactions?

**Old gold text** (2c25b162fb2c, adverse_reactions):
> 6 ADVERSE REACTIONS The following serious adverse reactions are discussed in greater detail in other sections of the labeling: Infusion Reactions [see Warnings and Precautions (5.1)] Infections [see Warnings and Precautions (5.2)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.3)] Reduction in Immunoglobulins [see Warnings and Precautions (5.4)] Malignancies [see Warnings and Precautions (5.5)] Immune-Mediated Colitis [see Warnings and Precautions (5.6)] Liver Injury

**A** `2c25b162fb2c` · Ocrevus · adverse_reactions · chunk 1/7
> 6 ADVERSE REACTIONS The following serious adverse reactions are discussed elsewhere in the labeling: Infusion Reactions [see Warnings and Precautions (5.1)] Infections [see Warnings and Precautions (5.2)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.3)] Reduction in Immunoglobulins [see Warnings and Precautions (5.4)] Malignancies [see Warnings and Precautions (5.5)] Immune-Mediated Colitis [see Warnings and Precautions (5.6)] Liver Injury [see Warnings and Precaut

**B** `03201e9e5b0b` · Ocrevus · adverse_reactions · chunk 6/7
> cause these reactions are reported voluntarily from a population of uncertain size, it is not always possible to reliably estimate their frequency or establish a causal relationship to drug exposure. Gastrointestinal Disorders: Immune-mediated colitis [see Warnings and Precautions (5.6)] Hepatobiliary Disorders: Liver injury [see Warnings and Precautions (5.7)] Infections and Infestations: Serious herpes infections [see Warnings and Precautions (5.2)], progressive multifocal leukoencephalopathy 

**C** `72ede922849e` · Ocrevus · adverse_reactions · chunk 7/7
> Infestations: Serious herpes infections [see Warnings and Precautions (5.2)], progressive multifocal leukoencephalopathy [see Warnings and Precautions (5.3)], and babesiosis Skin: Pyoderma gangrenosum


---
## WP-002  (Tysabri · expected section: boxed_warning)

**Question:** What is the black box warning for natalizumab?

**Old gold text** (af30f0f2afbb, warnings_and_precautions):
> 5 WARNINGS AND PRECAUTIONS •Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYRUKO if these infections occur and treat appropriately. (5.3) •Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYRUKO in patients with evidence of liver injury. (5.4) •Hypersensitivity reactions: Serious 

**A** `03b4ce4527bd` · Tyruko · warnings_and_precautions · chunk 1/14
> 5 WARNINGS AND PRECAUTIONS •Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYRUKO if these infections occur and treat appropriately. (5.3) •Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYRUKO in patients with evidence of liver injury. (5.4) •Hypersensitivity reactions: Serious 

**B** `af30f0f2afbb` · Tysabri · warnings_and_precautions · chunk 1/14
> 5 WARNINGS AND PRECAUTIONS Herpes infections: Life-threatening and fatal cases have occurred with herpes encephalitis and meningitis infections. Blindness has occurred in patients developing acute retinal necrosis. Discontinue TYSABRI if these infections occur and treat appropriately (5.3) Hepatotoxicity: Significant liver injury, including liver failure requiring transplant, has occurred. Discontinue TYSABRI in patients with evidence of liver injury (5.4) Hypersensitivity reactions: Serious hyp

**C** `b7c71cf92203` · Tyruko · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY Natalizumab products increase the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with T

**D** `59ed4675b125` · Tysabri · boxed_warning · chunk 1/3
> WARNING: PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY TYSABRI increases the risk of progressive multifocal leukoencephalopathy (PML), an opportunistic viral infection of the brain that usually leads to death or severe disability. Risk factors for the development of PML include the presence of anti-JCV antibodies, duration of therapy, and prior use of immunosuppressants. These factors should be considered in the context of expected benefit when initiating and continuing treatment with TYSABRI [ see

**E** `1fd0911a16fb` · Tysabri · boxed_warning · chunk 2/3
> r patients, and withhold TYSABRI immediately at the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**F** `71a9166425dc` · Tysabri · boxed_warning · chunk 3/3
> the first sign or symptom suggestive of PML (4, 5.1 ) Because of the risk of PML, TYSABRI is available only through a restricted distribution program called the TOUCH ® Prescribing Program (5.1, 5.2)

**G** `6c082b50694f` · Tyruko · boxed_warning · chunk 2/3
> he context of expected benefit when initiating and continuing treatment with TYRUKO (5.1). • Monitor patients, and withhold TYRUKO immediately at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).

**H** `826d818c05d3` · Tyruko · boxed_warning · chunk 3/3
> ely at the first sign or symptom suggestive of PML (4, 5.1). • Because of the risk of PML, TYRUKO is available only through a restricted distribution program called the TYRUKO REMS Program (5.1, 5.2).


---
## WP-003  (Gilenya · expected section: warnings_and_precautions)

**Question:** What are the cardiac risks of fingolimod?

**Old gold text** (37aac46053e9, warnings_and_precautions):
> 5 WARNINGS AND PRECAUTIONS Infections: Fingolimod may increase the risk. Obtain a complete blood count (CBC) before initiating treatment. Monitor for infection during treatment and for 2 months after discontinuation. Do not start in patients with active infections. (5.2) Progressive Multifocal Leukoencephalopathy (PML): Withhold fingolimod at the first sign or symptom suggestive of PML. (5.3) Macular Edema: Increases the risk of macular edema. Obtain a baseline evaluation of the fundus, includin

**A** `37aac46053e9` · Gilenya · warnings_and_precautions · chunk 1/17
> 5 WARNINGS AND PRECAUTIONS Infections: GILENYA may increase the risk. Obtain a complete blood count (CBC) before initiating treatment. Monitor for infection during treatment and for 2 months after discontinuation. Do not start in patients with active infections. (5.2) Progressive Multifocal Leukoencephalopathy (PML): Withhold GILENYA at the first sign or symptom suggestive of PML. (5.3) Macular Edema: Increases the risk of macular edema. Obtain a baseline evaluation of the fundus, including the 

**B** `934dcb31d207` · Gilenya · dosage_and_administration · chunk 2/5
> AV) conduction [see Dosage and Administration (2.4), Drug Interactions (7.5)]. Complete Blood Count (CBC) Review results of a recent CBC [see Warnings and Precautions (5.2), Drug Interactions (7.6)]. Serum Transaminases (ALT and AST) and Total Bilirubin Levels Prior to starting treatment with GILENYA (i.e., within 6 months), obtain serum transaminases [alanine transaminase (ALT) and aspartate transferase (AST)] and total bilirubin levels [see Warnings and Precautions (5.5)]. Ophthalmic Assessmen

**C** `1345508d8a34` · Gilenya · drug_interactions · chunk 1/3
> 7 DRUG INTERACTIONS Systemic Ketoconazole: Monitor during concomitant use. (7.2, 12.3) Vaccines: Avoid live attenuated vaccines during, and for 2 months after stopping GILENYA treatment. (5.2, 7.3) 7.1 QT Prolonging Drugs GILENYA has not been studied in patients treated with drugs that prolong the QT interval. Drugs that prolong the QT interval have been associated with cases of torsades de pointes in patients with bradycardia. Since initiation of GILENYA treatment results in decreased heart rat


---
## WP-006  (Gilenya · expected section: warnings_and_precautions)

**Question:** What are the macular edema risks with fingolimod?

**Old gold text** (73ce7f0f4772, warnings_and_precautions):
> ors, including fingolimod, have been associated with an increased risk of macular edema. Obtain a baseline evaluation of the fundus, including the macula, near the start of treatment with fingolimod. Perform an examination of the fundus, including the macula, 3 to 4 months after starting treatment, periodically while on therapy, and any time there is a change in vision. A dose-dependent increase in the risk of macular edema occurred in the fingolimod clinical development program. In 2-year doubl

**A** `73ce7f0f4772` · Gilenya · warnings_and_precautions · chunk 10/17
> of treatment with GILENYA. Perform an examination of the fundus, including the macula, 3 to 4 months after starting treatment, periodically while on therapy, and any time there is a change in vision. A dose-dependent increase in the risk of macular edema occurred in the GILENYA clinical development program. In 2-year double-blind, placebo-controlled studies in adult patients with multiple sclerosis, macular edema with or without visual symptoms occurred in 1.5% of patients (11/799) treated with 

**B** `9fadb7153686` · Gilenya · warnings_and_precautions · chunk 11/17
> if macular edema develops; this decision should include an assessment of the potential benefits and risks for the individual patient. The risk of recurrence after rechallenge has not been evaluated. Macular Edema in Patients with History of Uveitis or Diabetes Mellitus Patients with a history of uveitis and patients with diabetes mellitus are at increased risk of macular edema during GILENYA therapy. In the combined clinical trial experience in adult patients with all doses of fingolimod, the ra

**C** `5eaf24b41b55` · Gilenya · warnings_and_precautions · chunk 4/17
> rns to baseline. Physicians should continue to be alert to patient reports of cardiac symptoms. Atrioventricular Blocks Initiation of GILENYA treatment has resulted in transient AV conduction delays. In controlled clinical trials in adult patients, first-degree AV block after the first dose occurred in 4.7% of patients receiving GILENYA and 1.6% of patients on placebo. In a study of 697 patients with available 24-hour Holter monitoring data after their first dose (N = 351 receiving GILENYA and N


---
## DA-001  (Ocrevus · expected section: dosage_and_administration)

**Question:** What is the recommended dose for ocrelizumab?

**Old gold text** (fb8fe3c47b58, dosage_and_administration):
> r live or live-attenuated vaccines and, whenever possible, at least 2 weeks prior to initiation of OCREVUS for non-live vaccines [see Warnings and Precautions (5.2) and Clinical Pharmacology (12.2)]. Liver Function Tests Prior to initiating OCREVUS, obtain serum aminotransferases (alanine aminotransferase [ALT] and aspartate aminotransferase [AST]), alkaline phosphatase, and bilirubin levels [see Warnings and Precautions (5.7)]. 2.2 Preparation Before Every Infusion Infection Assessment Prior to

**A** `fb8fe3c47b58` · Ocrevus · dosage_and_administration · chunk 2/7
> testing for quantitative serum immunoglobulins [see Warnings and Precautions (5.4)]. For patients with low serum immunoglobulins, consult immunology experts before initiating treatment with OCREVUS. Vaccinations Because vaccination with live-attenuated or live vaccines is not recommended during treatment and after discontinuation until B-cell repletion, administer all age-appropriate immunizations according to immunization guidelines at least 4 weeks prior to initiation of OCREVUS for live or li

**B** `03201e9e5b0b` · Ocrevus · adverse_reactions · chunk 6/7
> cause these reactions are reported voluntarily from a population of uncertain size, it is not always possible to reliably estimate their frequency or establish a causal relationship to drug exposure. Gastrointestinal Disorders: Immune-mediated colitis [see Warnings and Precautions (5.6)] Hepatobiliary Disorders: Liver injury [see Warnings and Precautions (5.7)] Infections and Infestations: Serious herpes infections [see Warnings and Precautions (5.2)], progressive multifocal leukoencephalopathy 

**C** `72ede922849e` · Ocrevus · adverse_reactions · chunk 7/7
> Infestations: Serious herpes infections [see Warnings and Precautions (5.2)], progressive multifocal leukoencephalopathy [see Warnings and Precautions (5.3)], and babesiosis Skin: Pyoderma gangrenosum


---
## DA-002  (Tysabri · expected section: dosage_and_administration)

**Question:** How is natalizumab administered?

**Old gold text** (55dd689b658f, dosage_and_administration):
> 2 DOSAGE AND ADMINISTRATION •300 mg infused intravenously over one hour, every four weeks. Do not give as an intravenous push or bolus. (2.1, 2.2) •TYRUKO solution must be administered within 4 hours of preparation. (2.3) •Observe patients during all infusions. Post-infusion, for the first 12 infusions, observe patients for one hour after the infusion is complete. For patients who have received 12 infusions without evidence of a hypersensitivity reaction, observe patients post-infusion for the 1

**A** `f80809775d2b` · Tyruko · dosage_and_administration · chunk 1/4
> 2 DOSAGE AND ADMINISTRATION •300 mg infused intravenously over one hour, every four weeks. Do not give as an intravenous push or bolus. (2.1, 2.2) •TYRUKO solution must be administered within 4 hours of preparation. (2.3) •Observe patients during all infusions. Post-infusion, for the first 12 infusions, observe patients for one hour after the infusion is complete. For patients who have received 12 infusions without evidence of a hypersensitivity reaction, observe patients post-infusion for the 1

**B** `55dd689b658f` · Tysabri · dosage_and_administration · chunk 1/4
> 2 DOSAGE AND ADMINISTRATION 300 mg infused intravenously over one hour, every four weeks. Do not give as an intravenous push or bolus (2.1, 2.2) TYSABRI solution must be administered within 48 hours of preparation (2.3) Observe patients during all infusions. Post-infusion, for the first 12 infusions, observe patients for one hour after the infusion is complete. For patients who have received 12 infusions without evidence of a hypersensitivity reaction, observe patients post-infusion for the 13th

**C** `d9bddc871cdf` · Tyruko · dosage_and_administration · chunk 3/4
> mediately, or refrigerate the diluted solution at 2°C to 8°C, and use within 4 hours. If stored at 2°C to 8°C, allow the diluted solution to warm to room temperature prior to infusion. DO NOT FREEZE. 2.4 Administration Instructions •Infuse TYRUKO 300 mg in 100 mL 0.9% Sodium Chloride Injection, USP, over approximately one hour (infusion rate approximately 5 mg per minute). Do not administer TYRUKO as an intravenous push or bolus injection. After the infusion is complete, flush with 0.9% Sodium C


---
## DA-004  (Gilenya · expected section: dosage_and_administration)

**Question:** How should fingolimod be initiated?

**Old gold text** (9f2254c44f67, dosage_and_administration):
> iscontinuation for longer than 14 days, require first-dose monitoring. This monitoring is also recommended when the dose is increased in pediatric patients [see Dosage and Administration (2.4, 2.5)]. Fingolimod capsules can be taken with or without food. 2.3 Recommended Dosage In adults and pediatric patients 10 years of age and older weighing more than 40 kg, the recommended dosage of fingolimod capsules is 0.5 mg orally once-daily. In pediatric patients 10 years of age and older weighing less 

**A** `9f2254c44f67` · Gilenya · dosage_and_administration · chunk 3/5
> uire first-dose monitoring. This monitoring is also recommended when the dose is increased in pediatric patients [see Dosage and Administration (2.4, 2.5)]. GILENYA can be taken with or without food. 2.3 Recommended Dosage In adults and pediatric patients 10 years of age and older weighing more than 40 kg, the recommended dosage of GILENYA is 0.5 mg orally once daily. In pediatric patients 10 years of age and older weighing less than or equal to 40 kg, the recommended dosage of GILENYA is 0.25 m

**B** `16b09b5f8eef` · Gilenya · indications_and_usage · chunk 1/1
> 1 INDICATIONS AND USAGE GILENYA is indicated for the treatment of relapsing forms of multiple sclerosis (MS), to include clinically isolated syndrome, relapsing-remitting disease, and active secondary progressive disease, in patients 10 years of age and older. GILENYA is a sphingosine 1-phosphate receptor modulator indicated for the treatment of relapsing forms of multiple sclerosis (MS), to include clinically isolated syndrome, relapsing-remitting disease, and active secondary progressive disea

**C** `ec6d0d386a8b` · Gilenya · use_in_specific_populations · chunk 5/6
> highest dose tested and in males at all doses. The bone changes observed in fingolimod-treated juvenile rats are consistent with a reported role of S1P in the regulation of bone mineral homeostasis. When fingolimod (0.5 or 5 mg/kg/day) was orally administered to rats from the neonatal period through sexual maturity, a marked decrease in T-cell dependent antibody response was observed at both doses. This effect had not fully recovered by 6 to 8 weeks after the end of treatment. Overall, a no-effe


---
## DI-001  (Gilenya · expected section: drug_interactions)

**Question:** Does fingolimod interact with beta blockers?

**Old gold text** (c85cc8670fc7, drug_interactions):
> ncluding corticosteroids) are expected to increase the risk of immunosuppression, and the risk of additive immune system effects must be considered if these therapies are coadministered with fingolimod. When switching from drugs with prolonged immune effects, such as natalizumab, teriflunomide or mitoxantrone, the duration and mode of action of these drugs must be considered to avoid unintended additive immunosuppressive effects when initiating fingolimod [see Warnings and Precautions (5.2)]. 7.

**A** `c85cc8670fc7` · Gilenya · drug_interactions · chunk 2/3
> ncluding corticosteroids) are expected to increase the risk of immunosuppression, and the risk of additive immune system effects must be considered if these therapies are coadministered with GILENYA. When switching from drugs with prolonged immune effects, such as natalizumab, teriflunomide or mitoxantrone, the duration and mode of action of these drugs must be considered to avoid unintended additive immunosuppressive effects when initiating GILENYA [see Warnings and Precautions (5.2)]. 7.5 Drug

**B** `934dcb31d207` · Gilenya · dosage_and_administration · chunk 2/5
> AV) conduction [see Dosage and Administration (2.4), Drug Interactions (7.5)]. Complete Blood Count (CBC) Review results of a recent CBC [see Warnings and Precautions (5.2), Drug Interactions (7.6)]. Serum Transaminases (ALT and AST) and Total Bilirubin Levels Prior to starting treatment with GILENYA (i.e., within 6 months), obtain serum transaminases [alanine transaminase (ALT) and aspartate transferase (AST)] and total bilirubin levels [see Warnings and Precautions (5.5)]. Ophthalmic Assessmen

**C** `e89057faa937` · Gilenya · adverse_reactions · chunk 1/6
> 6 ADVERSE REACTIONS The following serious adverse reactions are described elsewhere in labeling: Bradyarrhythmia and Atrioventricular Blocks [see Warnings and Precautions (5.1)] Infections [see Warnings and Precautions (5.2)] Progressive Multifocal Leukoencephalopathy [see Warnings and Precautions (5.3)] Macular Edema [see Warnings and Precautions (5.4)] Liver Injury [see Warnings and Precautions (5.5)] Posterior Reversible Encephalopathy Syndrome [see Warnings and Precautions (5.6)] Respiratory


---
## SP-002  (Gilenya · expected section: use_in_specific_populations)

**Question:** Can fingolimod be used in pediatric patients?

**Old gold text** (ca574a4b6b27, use_in_specific_populations):
> on the potential for a serious risk to the fetus and the need for effective contraception during treatment with fingolimod [see Warnings and Precautions (5.8) and Use in Specific Populations (8.1)]. Since it takes approximately 2 months to eliminate the compound from the body after stopping treatment, the potential risk to the fetus may persist and women should use effective contraception during this period [see Warnings and Precautions (5.8, 5.13)]. 8.4 Pediatric Use Safety and effectiveness of

**A** `ca574a4b6b27` · Gilenya · use_in_specific_populations · chunk 4/6
> led on the potential for a serious risk to the fetus and the need for effective contraception during treatment with GILENYA [see Warnings and Precautions (5.8) and Use in Specific Populations (8.1)]. Since it takes approximately 2 months to eliminate the compound from the body after stopping treatment, the potential risk to the fetus may persist and women should use effective contraception during this period [see Warnings and Precautions (5.8, 5.13)]. 8.4 Pediatric Use Safety and effectiveness o

**B** `c7a9bd896ab6` · Gilenya · adverse_reactions · chunk 4/6
> als and in the postmarketing setting in adults [see Adverse Reactions (6.2)]. In adult clinical trials, the rate of seizures was 0.9% in GILENYA-treated patients and 0.3% in placebo-treated patients. It is unknown whether these events were related to the effects of multiple sclerosis alone, to GILENYA, or to a combination of both. Pediatric Patients 10 Years of Age and Older In the controlled pediatric trial (Study 4), the safety profile in pediatric patients receiving GILENYA 0.25 mg or 0.5 mg 

**C** `e8eebaaff03e` · Gilenya · adverse_reactions · chunk 6/6
> uding mycosis fungoides), Kaposi’s sarcoma, squamous cell carcinoma [see Warnings and Precautions (5.12)] Skin and Subcutaneous Tissue Disorders: Hypersensitivity [see Warnings and Precautions (5.14)]


---
## SP-005  (Mayzent · expected section: use_in_specific_populations)

**Question:** Can elderly patients take siponimod?

**Old gold text** (d65f1c1e41f1, use_in_specific_populations):
> breastfeeding should be considered along with the mother’s clinical need for MAYZENT and any potential adverse effects on the breastfed infant from MAYZENT or from the underlying maternal condition. 8.3 Females and Males of Reproductive Potential Contraception Females Before initiation of MAYZENT treatment, women of childbearing potential should be counseled on the potential for a serious risk to the fetus and the need for effective contraception during treatment with MAYZENT [see Use in Specifi

**A** `d65f1c1e41f1` · Mayzent · use_in_specific_populations · chunk 3/5
> breastfeeding should be considered along with the mother’s clinical need for MAYZENT and any potential adverse effects on the breastfed infant from MAYZENT or from the underlying maternal condition. 8.3 Females and Males of Reproductive Potential Contraception Females Before initiation of MAYZENT treatment, women of childbearing potential should be counseled on the potential for a serious risk to the fetus and the need for effective contraception during treatment with MAYZENT [see Use in Specifi

**B** `4f158aae2c1c` · Mayzent · drug_interactions · chunk 4/4
> not recommended. Monitor for loss of efficacy with concomitant use of MAYZENT and moderate or strong CYP3A4 inducers in patients with CYP2C9*1/*3 or *2/*3 genotype [see Clinical Pharmacology (12.3)].

**C** `cec70a409b69` · Mayzent · dosage_and_administration · chunk 1/6
> 2 DOSAGE AND ADMINISTRATION Assessments are required prior to initiating MAYZENT. (2.1) Titration is required for treatment initiation. (2.2, 2.3) The recommended maintenance dosage is 2 mg. (2.2) The recommended maintenance dosage in patients with a CYP2C9*1/*3 or *2/*3 genotype is 1 mg. (2.3) Administer tablets whole; do not split, crush, or chew. (2.2, 2.3) First-dose monitoring is recommended for patients with sinus bradycardia, first- or second-degree [Mobitz type I] atrioventricular (AV) b


---
## IU-002  (Tysabri · expected section: indications_and_usage)

**Question:** What type of MS is natalizumab used for?

**Old gold text** (8d8b216260aa, indications_and_usage):
> 1 INDICATIONS AND USAGE TYRUKO is an integrin receptor antagonist indicated for treatment of: Multiple Sclerosis (MS) TYRUKO is indicated as monotherapy for the treatment of relapsing forms of multiple sclerosis, to include clinically isolated syndrome, relapsing-remitting disease, and active secondary progressive disease, in adults. Natalizumab products increase the risk of PML [see Warnings and Precautions (5.1)]. When initiating and continuing treatment with TYRUKO, physicians should consider

**A** `46fd32aae5b7` · Tyruko · indications_and_usage · chunk 1/3
> 1 INDICATIONS AND USAGE TYRUKO is an integrin receptor antagonist indicated for treatment of: Multiple Sclerosis (MS) TYRUKO is indicated as monotherapy for the treatment of relapsing forms of multiple sclerosis, to include clinically isolated syndrome, relapsing-remitting disease, and active secondary progressive disease, in adults. Natalizumab products increase the risk of PML [see Warnings and Precautions (5.1)]. When initiating and continuing treatment with TYRUKO, physicians should consider

**B** `8d8b216260aa` · Tysabri · indications_and_usage · chunk 1/3
> 1 INDICATIONS AND USAGE TYSABRI is an integrin receptor antagonist indicated for treatment of: Multiple Sclerosis (MS) TYSABRI is indicated as monotherapy for the treatment of relapsing forms of multiple sclerosis, to include clinically isolated syndrome, relapsing-remitting disease, and active secondary progressive disease, in adults. TYSABRI increases the risk of PML [See Warnings and Precautions ( 5.1)]. When initiating and continuing treatment with TYSABRI, physicians should consider whether

**C** `f5ead356b4a4` · Tyruko · warnings_and_precautions · chunk 14/14
> on the effects of vaccination in patients receiving natalizumab products. No data are available on the secondary transmission of infection by live vaccines in patients receiving natalizumab products.


---
## PT-005  (Zeposia · expected section: contraindications)

**Question:** Can I take ozanimod if I have heart problems?

**Old gold text** (64f3839386fa, contraindications):
> 4 CONTRAINDICATIONS ZEPOSIA is contraindicated in patients who: •In the last 6 months, have experienced a myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization, or Class III or IV heart failure [see Warnings and Precautions (5.3) ] •Have the presence of Mobitz type II second-degree or third degree atrioventricular (AV) block, sick sinus syndrome, or sino-atrial block, unless the patient has a functioning pacemaker [

**A** `64f3839386fa` · Zeposia · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS ZEPOSIA is contraindicated in patients who: In the last 6 months, have experienced a myocardial infarction, unstable angina, stroke, transient ischemic attack (TIA), decompensated heart failure requiring hospitalization, or Class III or IV heart failure [see Warnings and Precautions (5.3) ] Have the presence of Mobitz type II second-degree or third degree atrioventricular (AV) block, sick sinus syndrome, or sino-atrial block, unless the patient has a functioning pacemaker [se

**B** `9e278dfabb0b` · Zeposia · use_in_specific_populations · chunk 5/5
> and its active metabolites were not evaluated in patients with severe hepatic impairment (Child-Pugh class C). Therefore, use of ZEPOSIA in patients with severe hepatic impairment is not recommended.

**C** `9805a2de2126` · Zeposia · dosage_and_administration · chunk 1/4
> 2 DOSAGE AND ADMINISTRATION Assessments are required prior to initiating ZEPOSIA. ( 2.1) Titration is required for treatment initiation. ( 2.2) The recommended maintenance dosage is 0.92 mg orally once daily. ( 2.2) The recommended maintenance dosage in patients with mild or moderate chronic hepatic impairment (Child-Pugh class A or B) is 0.92 mg once every other day. ( 2.3) If a dose is missed within the first 2 weeks of treatment, reinitiate with the titration regimen. If a dose is missed afte


---
## MD-002  (Ocrevus · expected section: dosage_and_administration)

**Question:** Compare the infusion requirements of ocrelizumab and natalizumab.

**Old gold text** (c87349f71b0e, dosage_and_administration):
> ous infusion. Subsequent doses: single 600 mg intravenous infusion every 6 months. Observe the patient for at least one hour after the completion of the infusion [see Warnings and Precautions (5.1)]. Table 1 Recommended Dose, Infusion Rate, and Infusion Duration for RMS and PPMS Amount and Volume Solutions of OCREVUS for intravenous infusion are prepared by dilution of the drug product into an infusion bag containing 0.9% Sodium Chloride Injection, to a final drug concentration of approximately 

**A** `c87349f71b0e` · Ocrevus · dosage_and_administration · chunk 3/7
> hydramine) 30 to 60 minutes prior to each OCREVUS infusion. An antipyretic (e.g., acetaminophen) may also be considered. Administer the antipyretic 30 minutes to 60 minutes prior to OCREVUS infusion. 2.3 Recommended Dosage and Dose Administration Administer OCREVUS under the close supervision of an experienced healthcare professional with access to appropriate medical support to manage severe reactions such as serious infusion reactions. Observe the patient for at least one hour after the comple

**B** `8530fbe8efa0` · Ocrevus · dosage_and_administration · chunk 4/7
> mL per hour every 30 minutes Maximum: 200 mL per hour Duration: 3.5 hours or longer OR Option 2(If no prior serious infusion reaction with any previous OCREVUS infusion) [see Adverse Reactions (6.1) and Clinical Studies (14.3)]. Infusion of approximately 2 hours duration 600 mg in 500 mL Start at 100 mL per hour for the first 15 minutes Increase to 200 mL per hour for the next 15 minutes Increase to 250 mL per hour for the next 30 minutes Increase to 300 mL per hour for the remaining 60 minutes 

**C** `72ede922849e` · Ocrevus · adverse_reactions · chunk 7/7
> Infestations: Serious herpes infections [see Warnings and Precautions (5.2)], progressive multifocal leukoencephalopathy [see Warnings and Precautions (5.3)], and babesiosis Skin: Pyoderma gangrenosum


---
## MD-005  (Copaxone · expected section: dosage_and_administration)

**Question:** Between glatiramer acetate and interferon beta-1a, which is injected?

**Old gold text** (581250a76c67, dosage_and_administration):
> 2 DOSAGE AND ADMINISTRATION For subcutaneous injection only; doses are not interchangeable (2.1) Glatiramer Acetate Injection 20 mg/mL per day (2.1) Glatiramer Acetate Injection 40 mg/mL three times per week (2.1) Before use, allow the solution to warm to room temperature (2.2) 2.1 Recommended Dose Glatiramer Acetate Injection is for subcutaneous use only [see Dosage and Administration (2.2)]. Do not administer intravenously. The dosing schedule depends on the product strength that is selected. 

**A** `581250a76c67` · Copaxone · dosage_and_administration · chunk 1/1
> 2 DOSAGE AND ADMINISTRATION For subcutaneous injection only; doses are not interchangeable (2.1) COPAXONE 20 mg/mL per day (2.1) COPAXONE 40 mg/mL three times per week (2.1) Before use, allow the solution to warm to room temperature (2.2) 2.1 Recommended Dose COPAXONE is for subcutaneous use only [see Dosage and Administration (2.2)]. Do not administer intravenously. The dosing schedule depends on the product strength that is selected. The recommended doses are: COPAXONE 20 mg per mL: administer

**B** `f9afbe6cc42b` · Rebif · dosage_and_administration · chunk 1/4
> 2 DOSAGE AND ADMINISTRATION For subcutaneous injection only (2.1) The recommended dose is either 22 mcg or 44 mcg injected subcutaneously three times per week (2.1) Titration: Generally, the starting dose should be 20% of the prescribed dose three times per week, and increased over a 4 week period to the targeted recommended dose of either 22 mcg or 44 mcg injected subcutaneously three times per week (2.1) Analgesics and/or antipyretics on treatment days may help ameliorate flu-like symptoms (2.

**C** `e08948ab597a` · Avonex · use_in_specific_populations · chunk 4/4
> not been established. 8.5 Geriatric Use Clinical studies of AVONEX did not include sufficient numbers of patients aged 65 and over to determine whether they respond differently than younger patients.


---
## MD-007  (Lemtrada · expected section: boxed_warning)

**Question:** Do alemtuzumab and cladribine both have boxed warnings for malignancy?

**Old gold text** (missing, ?):
> (chunk no longer exists)

**A** `19ebac0fe3cd` · Mavenclad · warnings_and_precautions · chunk 11/11
> indications other than multiple sclerosis. Instruct patients to seek medical advice if they experience symptoms of cardiac failure (e.g., shortness of breath, rapid or irregular heartbeat, swelling).

**B** `6aab139c06d4` · Lemtrada · adverse_reactions · chunk 7/7
> ommended in the treatment of MS. Cardiac Disorders: Congestive heart failure, cardiomyopathy, and decreased ejection fraction in non-MS patients previously treated with potentially cardiotoxic agents.

**C** `8ed7d7154e39` · Lemtrada · use_in_specific_populations · chunk 4/4
> .1, 5.2, 5.3, 5.4)]. 8.5 Geriatric Use Clinical studies of LEMTRADA did not include sufficient numbers of patients aged 65 and over to determine whether they respond differently than younger patients.

**D** `322240535fac` · Lemtrada · boxed_warning · chunk 1/3
> WARNING: AUTOIMMUNITY, INFUSION REACTIONS, STROKE, AND MALIGNANCIES LEMTRADA causes serious, sometimes fatal, autoimmune conditions such as immune thrombocytopenia and anti-glomerular basement membrane disease. Monitor complete blood counts with differential, serum creatinine levels, and urinalysis with urine cell counts before starting treatment and then at monthly intervals until 48 months after the last dose of LEMTRADA [see Warnings and Precautions (5.1)]. LEMTRADA causes serious and life-th

**E** `957e6faa60ad` · Lemtrada · boxed_warning · chunk 2/3
> full prescribing information for complete boxed warning. LEMTRADA causes serious, sometimes fatal, autoimmune conditions such as immune thrombocytopenia and anti-glomerular basement membrane disease. Monitor complete blood counts with differential, serum creatinine levels, and urinalysis with urine cell counts monthly until 48 months after the last dose. (5.1) LEMTRADA causes serious and life-threatening infusion reactions. LEMTRADA must be administered in a setting with appropriate equipment an

**F** `7dea2e82cd48` · Lemtrada · boxed_warning · chunk 3/3
> ancies, including thyroid cancer, melanoma, and lymphoproliferative disorders. Perform baseline and yearly skin exams. (5.4) LEMTRADA is available only through a restricted distribution program. (5.5)

**G** `bf949f670609` · Mavenclad · boxed_warning · chunk 1/1
> WARNING: MALIGNANCIES AND RISK OF TERATOGENICITY Malignancies Treatment with MAVENCLAD may increase the risk of malignancy. MAVENCLAD is contraindicated in patients with current malignancy. In patients with prior malignancy or with increased risk of malignancy, evaluate the benefits and risks of the use of MAVENCLAD on an individual patient basis. Follow standard cancer screening guidelines in patients treated with MAVENCLAD [see Contraindications (4) and Warnings and Precautions (5.1)]. Risk of


---
## CI-004  (Aubagio · expected section: contraindications)

**Question:** Is teriflunomide contraindicated in liver disease?

**Old gold text** (df97eb7ab288, contraindications):
> 4 CONTRAINDICATIONS Severe hepatic impairment (4, 5.1) Pregnancy (4, 5.2, 8.1) Hypersensitivity (4, 5.5) Current leflunomide treatment (4) AUBAGIO is contraindicated in/with: Patients with severe hepatic impairment [see Warnings and Precautions (5.1)]. Pregnant women and females of reproductive potential not using effective contraception. AUBAGIO may cause fetal harm [see Warnings and Precautions (5.2, 5.3) and Use in Specific Populations (8.1)]. Patients with a history of a hypersensitivity rea

**A** `df97eb7ab288` · Aubagio · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS Severe hepatic impairment (4, 5.1) Pregnancy (4, 5.2, 8.1) Hypersensitivity (4, 5.5) Current leflunomide treatment (4) AUBAGIO is contraindicated in/with: Patients with severe hepatic impairment [see Warnings and Precautions (5.1)]. Pregnant women and females of reproductive potential not using effective contraception. AUBAGIO may cause fetal harm [see Warnings and Precautions (5.2, 5.3) and Use in Specific Populations (8.1)]. Patients with a history of a hypersensitivity rea

**B** `556e950adfce` · Aubagio · drug_interactions · chunk 3/3
> consider reducing the dose of these drugs and monitor patients closely for signs and symptoms of increased exposures to the drugs while patients are taking AUBAGIO [see Clinical Pharmacology (12.3)].

**C** `af63f66479bf` · Aubagio · boxed_warning · chunk 3/3
> treatment and during an accelerated drug elimination procedure (4, 5.2, 5.3, 8.1, 8.3). Stop AUBAGIO and use an accelerated drug elimination procedure if the patient becomes pregnant (5.2, 5.3, 8.1).

**D** `29f65fde987e` · Aubagio · boxed_warning · chunk 1/3
> WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting [see Warnings and Precautions (5.1)]. Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the risk of severe liver injury. Obtain transaminase and bilirubin levels within 6 months before initiation of AUBAGIO therapy.

**E** `6e31876f5a0a` · Aubagio · boxed_warning · chunk 2/3
> ns (5.2, 5.3), Use in Specific Populations (8.1, 8.3), and Clinical Pharmacology (12.3)]. WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY See full prescribing information for complete boxed warning. Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting (5.1). Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the ris


---
## WP-007  (Mavenclad · expected section: boxed_warning)

**Question:** What are the warnings for cladribine?

**Old gold text** (37209c34da4c, warnings_and_precautions):
> 5 WARNINGS AND PRECAUTIONS Lymphopenia: Monitor lymphocyte counts before, during and after treatment. (5.3) Infections: Serious, including life-threatening and fatal infections, have occurred. Screen patients for active and latent infections; delay treatment until infection is fully resolved or controlled. Vaccination of patients seronegative to varicella zoster virus (VZV) is recommended prior to treatment. Vaccination of patients seropositive to VZV with zoster vaccine recombinant, adjuvanted,

**A** `37209c34da4c` · Mavenclad · warnings_and_precautions · chunk 1/11
> 5 WARNINGS AND PRECAUTIONS Lymphopenia: Monitor lymphocyte counts before, during and after treatment. (5.3) Infections: Serious, including life-threatening and fatal infections, have occurred. Screen patients for active and latent infections; delay treatment until infection is fully resolved or controlled. Vaccination of patients seronegative to varicella zoster virus (VZV) is recommended prior to treatment. Vaccination of patients seropositive to VZV with zoster vaccine recombinant, adjuvanted,

**B** `f9f1316d95fa` · Mavenclad · dosage_and_administration · chunk 5/5
> 2.6 Recommended Concomitant Medication Herpes Prophylaxis Administer anti-herpes prophylaxis in patients with lymphocyte counts less than 200 cells per microliter [see Warnings and Precautions (5.4)].

**C** `b0188a5d85e8` · Mavenclad · warnings_and_precautions · chunk 6/11
> rious herpes zoster infections occurred in 0.2% of MAVENCLAD-treated patients. Vaccination of patients who are seronegative for varicella zoster virus is recommended prior to initiation of MAVENCLAD. Administer live-attenuated or live vaccines at least 4 to 6 weeks prior to starting MAVENCLAD. Vaccination with zoster vaccine recombinant, adjuvanted is recommended for patients who are seropositive to VZV, either prior to or during MAVENCLAD treatment, including when their lymphocyte counts are le

**D** `bf949f670609` · Mavenclad · boxed_warning · chunk 1/1
> WARNING: MALIGNANCIES AND RISK OF TERATOGENICITY Malignancies Treatment with MAVENCLAD may increase the risk of malignancy. MAVENCLAD is contraindicated in patients with current malignancy. In patients with prior malignancy or with increased risk of malignancy, evaluate the benefits and risks of the use of MAVENCLAD on an individual patient basis. Follow standard cancer screening guidelines in patients treated with MAVENCLAD [see Contraindications (4) and Warnings and Precautions (5.1)]. Risk of


---
## SP-001  (Aubagio · expected section: use_in_specific_populations)

**Question:** Is teriflunomide safe during pregnancy?

**Old gold text** (ba10ad28e22e, use_in_specific_populations):
> 8 USE IN SPECIFIC POPULATIONS 8.1 Pregnancy Risk Summary AUBAGIO is contraindicated for use in pregnant women and females of reproductive potential not using effective contraception because of the potential for fetal harm based on animal data [see Contraindications (4) and Warnings and Precautions (5.2)]. In animal reproduction studies in rat and rabbit, oral administration of teriflunomide during organogenesis caused teratogenicity and embryolethality at plasma exposures (AUC) lower than that a

**A** `ba10ad28e22e` · Aubagio · use_in_specific_populations · chunk 1/7
> 8 USE IN SPECIFIC POPULATIONS 8.1 Pregnancy Risk Summary AUBAGIO is contraindicated for use in pregnant women and females of reproductive potential not using effective contraception because of the potential for fetal harm based on animal data [see Contraindications (4) and Warnings and Precautions (5.2)]. In animal reproduction studies in rat and rabbit, oral administration of teriflunomide during organogenesis caused teratogenicity and embryolethality at plasma exposures (AUC) lower than that a

**B** `556e950adfce` · Aubagio · drug_interactions · chunk 3/3
> consider reducing the dose of these drugs and monitor patients closely for signs and symptoms of increased exposures to the drugs while patients are taking AUBAGIO [see Clinical Pharmacology (12.3)].

**C** `af63f66479bf` · Aubagio · boxed_warning · chunk 3/3
> treatment and during an accelerated drug elimination procedure (4, 5.2, 5.3, 8.1, 8.3). Stop AUBAGIO and use an accelerated drug elimination procedure if the patient becomes pregnant (5.2, 5.3, 8.1).

**D** `29f65fde987e` · Aubagio · boxed_warning · chunk 1/3
> WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting [see Warnings and Precautions (5.1)]. Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the risk of severe liver injury. Obtain transaminase and bilirubin levels within 6 months before initiation of AUBAGIO therapy.

**E** `6e31876f5a0a` · Aubagio · boxed_warning · chunk 2/3
> ns (5.2, 5.3), Use in Specific Populations (8.1, 8.3), and Clinical Pharmacology (12.3)]. WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY See full prescribing information for complete boxed warning. Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting (5.1). Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the ris


---
## MD-003  (Aubagio · expected section: boxed_warning)

**Question:** Which has a boxed warning, teriflunomide or dimethyl fumarate?

**Old gold text** (df97eb7ab288, contraindications):
> 4 CONTRAINDICATIONS Severe hepatic impairment (4, 5.1) Pregnancy (4, 5.2, 8.1) Hypersensitivity (4, 5.5) Current leflunomide treatment (4) AUBAGIO is contraindicated in/with: Patients with severe hepatic impairment [see Warnings and Precautions (5.1)]. Pregnant women and females of reproductive potential not using effective contraception. AUBAGIO may cause fetal harm [see Warnings and Precautions (5.2, 5.3) and Use in Specific Populations (8.1)]. Patients with a history of a hypersensitivity rea

**A** `df97eb7ab288` · Aubagio · contraindications · chunk 1/1
> 4 CONTRAINDICATIONS Severe hepatic impairment (4, 5.1) Pregnancy (4, 5.2, 8.1) Hypersensitivity (4, 5.5) Current leflunomide treatment (4) AUBAGIO is contraindicated in/with: Patients with severe hepatic impairment [see Warnings and Precautions (5.1)]. Pregnant women and females of reproductive potential not using effective contraception. AUBAGIO may cause fetal harm [see Warnings and Precautions (5.2, 5.3) and Use in Specific Populations (8.1)]. Patients with a history of a hypersensitivity rea

**B** `556e950adfce` · Aubagio · drug_interactions · chunk 3/3
> consider reducing the dose of these drugs and monitor patients closely for signs and symptoms of increased exposures to the drugs while patients are taking AUBAGIO [see Clinical Pharmacology (12.3)].

**C** `af63f66479bf` · Aubagio · boxed_warning · chunk 3/3
> treatment and during an accelerated drug elimination procedure (4, 5.2, 5.3, 8.1, 8.3). Stop AUBAGIO and use an accelerated drug elimination procedure if the patient becomes pregnant (5.2, 5.3, 8.1).

**D** `29f65fde987e` · Aubagio · boxed_warning · chunk 1/3
> WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting [see Warnings and Precautions (5.1)]. Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the risk of severe liver injury. Obtain transaminase and bilirubin levels within 6 months before initiation of AUBAGIO therapy.

**E** `6e31876f5a0a` · Aubagio · boxed_warning · chunk 2/3
> ns (5.2, 5.3), Use in Specific Populations (8.1, 8.3), and Clinical Pharmacology (12.3)]. WARNING: HEPATOTOXICITY and EMBRYOFETAL TOXICITY See full prescribing information for complete boxed warning. Hepatotoxicity Clinically significant and potentially life-threatening liver injury, including acute liver failure requiring transplant, has been reported in patients treated with AUBAGIO in the postmarketing setting (5.1). Concomitant use of AUBAGIO with other hepatotoxic drugs may increase the ris
