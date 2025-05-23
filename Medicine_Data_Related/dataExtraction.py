import fitz  # PyMuPDF
import re
import json

def extract_medicine_data(pdf_path, output_json, pages):
   """
   Extracts medicine attributes from a PDF and saves them in a JSON file.
  
   :param pdf_path: Path to the PDF file
   :param output_json: Path to save the JSON file
   :param pages: List of page numbers to extract data from (1-based index)
   """
   doc = fitz.open(pdf_path)
  
   extracted_text = ""
   for page_num in pages:
       if page_num - 1 < len(doc):
           extracted_text += doc[page_num - 1].get_text("text") + "\n"
   doc.close()
  
   lines = extracted_text.split("\n")
  
   attributes = [
       "Dosage form and strength",
       "Indications",
       "Contraindications/Precautions",
       "Dosage schedule",
       "Adverse eﬀects",
       "Drug and food interaction"
   ]
  
   attribute_variants = {
        "Dosage form and strength": ["Dosage form and strength", "Dosage forms and strengths", "Dosage form", "Dosage forms", "Dose form and strength"],
        "Indications": ["Indications", "Indication"],
        "Contraindications/Precautions": ["Contraindications", "Contraindication", "Precautions", "Precaution"],
        "Dosage schedule": ["Dosage schedule", "Dosage schedules"],
        "Adverse eﬀects": ["Adverse eﬀects", "Adverse eﬀect", "Adverse effects", "Adverse effect", "Side effects", "Side effect"],
        "Drug and food interaction": ["Drug and food interaction", "Drug and food interactions", "Interactions", "Interaction","Drugs and food interaction", "Drug and foods interaction", "Drugs and foods interaction", "Drugs and foods interactions"]
   }
  
   def match_attribute(line):
       for key, variants in attribute_variants.items():
           if any(line.lower().startswith(variant.lower()) for variant in variants):
               return key
       return None
#    medicine_pattern = re.compile(r"^([A-Z\s\(\)/-]+)(?:\s*\(([^)]+)\))?$")
   medicine_pattern = re.compile(r"^([A-Z0-9\s\(\)/\-,.]+?)(?:\s*\(([^()]+)\))?$")
  
   data = []
   current_medicine = None
   current_alternate_name = None
   current_attribute = None
   current_data = {attr: "" for attr in attributes}
  
   for line in lines:
       stripped_line = line.strip()
       match = medicine_pattern.match(stripped_line)
      
       if match and match.group(1).strip().lower() in map(str.lower, medicine_list):
           if current_medicine:
               data.append({
                   "Medicine": current_medicine,
                   "Alternate Name": current_alternate_name,
                   **{attr: current_data[attr].strip() for attr in attributes}
               })
          
           current_medicine = match.group(1).strip()
           current_alternate_name = match.group(2).strip() if match.group(2) else None
           current_data = {attr: "" for attr in attributes}
           current_attribute = None
      
       else:
            matched_attr = match_attribute(stripped_line)
            if matched_attr:
               current_attribute = matched_attr
            #    current_data[current_attribute] = stripped_line[len(matched_attr):].strip()
               if not current_data[current_attribute]:
                    current_data[current_attribute] = stripped_line[len(matched_attr):].strip()
               
          
            elif current_medicine and current_attribute and stripped_line:
                    current_data[current_attribute] += "\n" + stripped_line

               
  
   if current_medicine:
       data.append({
           "Medicine": current_medicine,
           "Alternate Name": current_alternate_name,
           **{attr: current_data[attr].strip() for attr in attributes}
       })
       
  
   with open(output_json, "w", encoding="utf-8") as file:
       json.dump(data, file, indent=4, ensure_ascii=False)




# Example usage
pdf_file = "godFile.pdf"
output_json_file = "god_output.json"
doc = fitz.open(pdf_file)
pages_to_extract = (
    list(range(64, 82)) + list(range(85, 116)) + list(range(118, 138)) + 
    list(range(139, 146)) + list(range(147, 162)) + list(range(163, 174)) + 
    list(range(178, 218)) + list(range(232, 252)) + list(range(255, 306)) + 
    list(range(308, 320)) + list(range(322, 336)) + list(range(354, 382)) + 
    list(range(383, 386)) + list(range(396, 418)) + list(range(420, 432)) + 
    list(range(435, 450))
)

medicine_list = ['Aluminium hydroxide', 'Magnesium hydroxide', 'Atropine sulphate', 'Drotaverine', 'Flavoxate hydrochloride', 'Hyoscine butyl bromide', 'Mebeverine hydrochloride', 'Cimetidine', 'Esomeprazole', 'Famotidine', 'Lansoprazole', 'Omeprazole', 'Pantoprazole', 'Rabeprazole', 'Ranitidine', 'Sucralfate', 'Cyclizine', 'Dimenhydrinate', 'Domperidone', 'Metoclopramide', 'Ondansetron', 'Prochlorperazine', 'Promethazine', 'Diphenoxylate', 'Loperamide', 'Oral rehydration salts', 'Rifaximin', 'Zinc sulfate', 'Ispaghula husk', 'Docusate sodium', 'Liquid paraffin', 'Bisacodyl', 'Senna', 'Lactulose', 'Macrogol 3350', 'Magensium Hydroxide (Milk of Magnesia)', 'Magnesium sulphate', 'Polyethylene Glycol 3350', 'Adalimumab', 'Budesonide', 'Hydrocortisone acetate', 'Infliximab', 'Mesalazine', 'Sulfasalazine', 'Octreotide', 'Terlipressin', 'Ursodeoxycholic acid', 'Glyceryl trinatre (Nitriglycerine)', 'Isosorbide dinitrate', 'Isosobide mononitrates', 'Atenolol', 'Bisoprolol', 'Carvedilol', 'Metoprolol', 'Propranolol', 'Diltiazem', 'Verapamil', 'Adenosine', 'Amiodarone', 'Digoxin', 'Disopyramide', 'Isoprenaline', 'Lidocaine/ Lignocaine', 'Procainamide',   'Captopril', 'Enalapril', 'Lisinopril', 'Ramipril', 'Irbesartan', 'Losartan', 'Telmisartan', 'Valsartan', 'Labetalol', 'Nebivolol', 'Amlodipine', 'Felodipine', 'Nifedipine', 'Clonidine', 'Hydralazine hydrochloride', 'Methyldopa', 'Prazosin', 'Sodium nitroprusside', 'Tamsulosin', 'Terazosin','Phenylephrine', 'Vasopressin',  'Dobutamine',  'Dopamine', 'Epinephrine /Adrenaline', 'Furosemide',  'Milrinone', 'NOREPINEPHRINE/NORADRENALINE', 'Bosentan', 'Sildenafil', 'Atorvastatin', 'Cholestyramine Resins', 'Ezetimibe', 'Fenofibrate', 'Gemfibrozil', 'Nicotinic acid (Niacin)', 'Simvastatin', 'Rosuvastatin', 'Ferrous fumarate', 'Ferrous fumarate with folic acid', 'Ferrous gluconate', 'Ferrous sulphate', 'Ferrous sulfate with ascorbic acid', 'Ferrous sulfate with folic acid', 'Iron dextran', 'Folic acid', 'Epoetin alfa', 'Heparin (Unfractionated)', 'Low molecular weight heparins', 'Warfarin', 'Acenocoumarol','Apixaban', 'Bivalirudin', 'Dabigatran', 'Fondaparinux', 'Rivaroxaban', 'Aspirin', 'Clopidogrel', 'Prasugrel', 'Ticlopidine', 'Alteplase (rTPA; TISSUE-TYPE PLASMINOGEN ACTIVATOR)', 'Streptokinase', 'Tenecteplase', 'Urokinase', 'Ethamsylate', 'Tranexamic acid', 'Protamine sulfate', 'PHYTOMENADIONE (VITAMIN K₁)', 'Human albumin', 'Factor IX complex', 'Desmopressin', 'Polygeline', 'Calcium', 'Iodine',  'Phosphates', 'Sodium flouride', 'Zinc', 'Vitamin A (Retinol)', "VITAMIN D₃ (CALCITRIOL)","Vitamin E (Tocopherol)","Vitamin B₁ (Thiamine)","Vitamin B₂ (Riboflavin)","Vitamin B₃ (Niacin)","Vitamin B₆ (Pyridoxine)","Vitamin B₉ (Folic acid)","Vitamin B₁₂ (Cobalamin)",'Vitamin C (Ascorbic acid)','Acetazolamide', 'Amiloride hydrochloride', 'Chlorthalidone', 'Eplerenone', 'Furosemide', 'Hydrochlorothiazide', 'Indapamide', 'Mannitol', 'Metolazone', 'Spironolactone', 'Torsemide', 'Triamterene', 'Desmopressin', 'Vasopressin (ANTIDIURETIC HORMONE)', 'Nitrofurantoin', 'Bethanchol', 'Flavoxate', 'Mirabegron', 'Oxybutynin', 'Solifenacin', 'Tolteradin', 'Albumin', 'Glucose', 'Glucose with sodium chloride', 'Potassium chloride', "Ringer's Lactate",'Sodium bicarbonate', 'Sodium chloride', 'Ipratropium bromide', 'Bambuterol', 'Formoterol', 'Salbutamol', 'Salmeterol', 'Terbutaline sulphate', 'Beclomethasone dipropionate', 'Budesonide', 'Hydrocortisone', 'Montelukast', 'Zafirlukast', 'Sodium cromoglycate', 'Ephedrine hydrochloride', 'Aminophylline', 'Doxofylline', 'Theophylline', 'Codeine phosphate', 'Dextromethorphan', 'Noscapine', 'Pholcodine', 'Bromhexine', 'Carbocysteine', 'Acetylcysteine', 'Phenylephrine', 'Pseudoephedrine', 'Caffeine citrate', 'Carbamazepine', 'Clobazam', 'Clonazepam', 'Diazepam', 'Gabapentin', 'Lacosamide', 'Lamotrigine', 'Levetiracetam', 'Oxcarbazepine', 'Phenobarbital', 'Phenytoin', 'Topiramate', 'Valproic acid', 'Amantadine', 'Benztropine', 'Bromocriptine', 'Entacapone', 'Levodopa and carbidopa', 'Oxphenadrine hydrochloride', 'Pramipexole', 'Rasagiline', 'Ropinirole', 'Selegiline hydrochloride', 'TRIHEXYPHENIDYL HYDROCHLORIDE (BENZHEXOL)','Ergotamine tartarate', 'Rizatriptan', 'Sumatriptan', 'Flunarizine',  'Pregabalin', 'Buprenorphine', 'Codeine phosphate', 'Methadone hydrochloride', 'Morphine sulphate', 'PETHIDINE HYDROCHLORIDE (MEPERIDINE)','Tramadol hydrochloride', 'Naloxone', 'Naltrexone', 'Pentazocine', 'Mirtazapine',  'Escitalopram', 'Fluoxetine', 'Fluvoxamine', 'Paroxetine', 'Sertraline', 'Trazodone hydrochloride', 'Duloxetine', 'Venlafaxine', 'Amitriptyline', 'Clomipramine', 'DOSULEPIN (DOTHIEPIN)','Imipramine', 'Nortryptyline', 'Chlorpromazine', 'Fluphenazine decanoate', 'Haloperidol', 'Trifluoperazine', 'Amisulpride', 'Aripiprazole', 'Clozapine', 'Olanzapine', 'Quetiapine', 'Risperidone', 'Atomoxetine', 'Donepezil hydrochloride', 'Memantine hydrochloride', 'Rivastigmine',  'Disulfiram',  'Bupropion', 'Lithium',  'Alprazolam', 'Chlordiazepoxide',  'Lorazepam', 'Zolpidem', 'Allopurinol', 'Colchicine', 'Febuxostat', 'Probenecid', 'Sulfinpyrazone', 'Abatacept', 'Adalimumab', 'Anakinra', 'Etanercept', 'Infliximab', 'Leflunomide', 'Methotrexate', 'Penicillamine', 'Rituximab', 'Sulfasalazine', 'Tocilizumab', 'Tofacutinib', 'Aceclofenac', 'Aspirin', 'Diclofenac', 'Flurbiprofen', 'Ibuprofen', 'Indomethacin', 'Mefenamic acid', 'Naproxen', 'Paracetamol', 'Piroxicam', 'Celecoxib', 'Etoricoxib', 'Diacerein', 'Alendronate',  'Etidronate', 'Ibandronate', 'Pamidronate', 'Risedronate', 'Zoledronate', 'Baclofen', 'Dexmedetomidine','Tizanidine', 'Denosumab', 'Diloxanide furoate', 'Metronidazole', 'Secnidazole', 'Tinidazole', 'Cefaclor', 'Cefadroxil', 'Cefalexin (Cephalexin)', 'Cefazolin', 'Cefepime', 'Cefixime', 'Cefoperazone', 'Cefotaxime', 'Cefpodoxime', 'Ceftazidime', 'Ceftriaxone', 'Cefuroxime', 'Amoxicillin with CLAVULANATE', 'Ampicillin with SULBACTAM', 'Bacampicillin', 'BENZYL PENICILLIN (PENICILLIN G)','Carbenicillin', 'Cloxacillin', 'Flucloxacillin', 'PHENOXY METHYL PENICILLIN (PENICILLIN V','Piperacillin', 'PIPERACILLIN with TAZOBACTAM','Amikacin', 'Gentamicin', '10.2.2.2 CHLORAMPHENICOL', 'Clindamycin', 'Linezolid',  'Azithromycin', 'Clarithromycin', 'Erythromycin', 'Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin', 'Nalidixic acid', 'Norfloxacin', 'Ofloxacin', 'COTRIMOXAZOLE (SULPHAMETHOXAZOLE and TRIMETHOPRIM)', 'Doxycycline', 'Minocycline', 'Oxytetracycline', 'Tetracycline', 'Vancomycin', 'Amphotericin B', 'Clotrimazole', 'Fluconazole', 'Griseofulvin', 'Itraconazole', 'Ketoconazole', 'Nystatin', 'Albendazole', 'Mebendazole', 'Miltefosine', 'Niclosamide', 'Pentamidine', 'Piperazine', 'Praziquantel', 'Pyrantel pamoate', 'Sodium stibogluconate', 'Clofazimine', 'Dapsone', 'Artemether with lumefantrine', 'Artesunate', 'Chloroquine', 'Mefloquine', 'Primaquine', 'Quinine', 'Sulfadoxine and pyrimethamine', 'Ethambutol', 'Isoniazid', 'Pyrazinamide', 'Rifampin', 'Streptomycin', 'Bedaquiline', 'Capreomycin', 'Cycloserine', 'Ethionamide', 'Kanamycin', 'P-AMINOSALICYLIC ACID (PAS)','Entecavir', 'PEGINTERFERON ALFA (POLYETHYLENE GLYCOL-CONJUGATED DERIVATIVES OF INTERFERON ALFA)', 'Sofosbuvir', 'Acyclovir', 'Oseltamivir', 'Didanosine', 'Efavirenz', 'Indinavir', 'Lamivudine', 'LOPINAVIR + RITONAVIR', 'Nelfinavir', 'Nevirapine', 'Ritonavir', 'Saquinavir', 'Stavudine', 'Tenofovir disoproxil', 'Zidovudine',  'Beclomethasone', 'Betamethasone', 'Cortisone acetate', 'Dexamethasone', 'Fludrocortisone acetate', 'Hydrocortisone', 'Methyl prednisolone', 'Prednisolone', 'Triamcinolone acetonide', 'Insulin aspart', 'Insulin glargine', 'INSULIN ISOPHANE (NPH)','Insulin lispro', 'Insulin protamine zinc (PZI)', 'Insulin soluble','INSULIN ZINC SUSPENSION', 'Acarbose', 'Metformin', 'Sitagliptin', 'Liraglutide', 'Repaglinide', 'Chlorpropamide', 'Glibenclamide', 'Gliclazide', 'GLIMEPERIDE','Glipizide', 'Pioglitazone', 'GLUCAGON (GLUCAGON HYDROCHLORIDE)', 'LEVOTHYROXINE (L-THYROXINE)',  'Carbimazole', 'Iodine', 'LUGOL’S IODINE', 'Propanolol', 'Propylthiouracil', 'Octreotide', 'Somatotropin', 'Calcitonin', 'SODIUM ALDRONATE', 'Teriparatide', 'Orlistat', 'Mesterolone', 'Methyl-testosterone', 'Testosterone', 'Nandrolone', 'Oxymetholone', 'Stanozolol', 'Bicalutamide', 'Danazol', 'Finasetride', 'Conjugated estrogen', 'Estradiol', 'Ethinylestradiol', 'Mestranol', 'Chorionic gonadotrophin', 'Clomiphene', 'Tamoxifen', 'Dydrogesterone', 'Hydroxyprogesterone', 'Medroxyprogesterone', 'Norethisterone', 'Combined oral contraceptive pills (OCPs)', 'Medroxyprogesterone', 'LEVONORGESTREL IMPLANT', 'LEVONORGESTREL', 'ERGOT ALKALOIDS (Ergometrine, Methyl ergometrine)' 'Mifepristone', 'MIFEPRISTONE with MISOPROSTOL','Oxytocin', 'PROSTAGLANDIN ANALOGUE', 'Nifedipine', 'Isoxsuprine', 'Salbutamol', 'Terbutaline', 'Actinomycin D (Dactinomycin)', 'Arsenic trioxide', 'Azathioprine', 'CALCIUM FOLINATE (CALCIUM LEUCOVORIN)','Capecitabine', 'Carboplatin', 'Chlorambucil', 'Cisplatin', 'Cyclophosphamide', 'Cytarabine', 'Daunorubicin', 'Docetaxel', 'Doxorubicin', 'Epirubicin', 'Etoposide', 'Fludarabine', 'Fluorouracil', 'Hydroxyurea', 'Irinotecan', 'Interferon beta', 'INTERFERON GAMMA-1B','Lomustine', 'Melphalan','MERCAPTOPURINE (6-MERCAPTOPURINE)','Methotrexate', 'Mitomycin', 'Mitoxantrone', 'Oxaliplatin', 'Paclitaxel', 'Pemetrexed', 'Procarbazine', 'Tacrolimus', 'Temozolomide', 'Topotecan', 'Tretinoin', 'Vincristine', 'Vinblastine', 'Vinorelbine', 'Bevacizumab', 'Bortezomib', 'Cetuximab', 'Erlotinib',  'Geftinib', 'Imatinib', 'Nilotinib', 'Osimertinib', 'Rituximab', 'Sunitinib', 'Trastuzumab', 'Anastrozole', 'Exemestane', 'Fulvestrant', 'Tamoxifen', 'TUBERCULIN, PURIFIED PROTEIN DERIVATIVE (PPD)','ANTI-D IMMUNOGLOBULIN (HUMAN)', 'Antirabies hyperimmune serum', 'Polyvenum antisnake serum', 'TETANUS IMMUNOGLOBULIN (HUMAN)', 'Benzoic acid and Salicylic acid', 'Fusidic Acid', 'Metronidazole', 'Mupirocin', 'Silver Sulfadiazine', 'Amorolfine', 'Clotrimazole', 'Gentian violet',  'Miconazole nitrate', 'Neomycin', 'Nystatin', 'Sertaconazole', 'Terbinafine', 'Benzyl Benzoate', 'Ivermectin', 'Permethrin', 'Aciclovir', 'Clobetasone butyrate', 'Clobetasol propionate', 'DITHRANOL (ANTHRALIN)','Flucinolone acetonide', 'Fluticasone', 'Hydrocortisone', 'Hydrocortisone butyrate', 'HYDROCORTISONE WITH FUSIDIC ACID','MOMETASONE FUROATE','Tacrolimus', 'Triamcinolone', 'Acitretin', 'Adapalene', 'Adapalene with Benzoyl peroxide', 'Isotretinoin', 'Tazarotene',  'Coal tar', 'Coal tar with Salicylic acid and Precipitated Sulfur', 'Calcipotriol', 'Calcitriol (1, 25-DIHYDROXYCHOLECALCIFEROL)', 'Aluminium chloride hexahydrate', 'GLYCOPYRRONIUM BROMIDE (GLYCOPYRROLATE)','Fluorouracil', 'Doxepin', 'Clindamycin', 'Azelaic acid', 'Benzoyl peroxide with Clindamycin', 'Erythromycin', 'Selenium sulphide', 'Minoxidil', 'Eflornithine', 'Chlorhexidine', 'Imiquimod', 'Potassium permanganate', 'Salicylic acid', 'Azelaic acid', 'Capsaicin', 'Hydroquinone', 'Psoralen', 'Chloramphenicol', 'Ciprofloxacin', 'Gentamicin', 'Levofloxacin', 'Neomycin', 'Ofloxacin', 'Polymyxin B', 'Tetracycline', 'Acyclovir', 'Ganciclovir', 'Idoxuridine', 'Fluconazole', 'Natamycin', 'Betamethasone', 'Diclofenac', 'Fluorometholone', 'Flurbiprofen', 'Hydrocortisone', 'Indomethacin', 'Ketorolac', 'Prednisolone', 'Ketotifen', 'Olopatadine', 'Sodium chromoglycate', 'Acetazolamide', 'Bimatoprost', 'Brimonidine', 'Dorzolamide', 'Latanoprost', 'Mannitol', 'Pilocarpine', 'Timolol',  'Travoprost', 'Atropine', 'Cyclopentolate', 'Homatropine', 'Phenylephrine', 'Tropicamide', 'Acetylcysteine', 'Bevacizumab', 'Bupivacaine', 'Carbomer', 'Cyclosporine', 'Fluorescein sodium', 'Hypromellose', 'Liquid paraffin', 'Polyvinyl alcohol', 'Sodium chloride', 'Triamcinolone', 'Chloramphenicol', 'Ciprofloxacin', 'Gentamicin', 'Ofloxacin', 'Clotrimazole', 'Chloramphenicol with Dexamethasone', 'Ciprofloxacin with Hydrocortisone', 'Neomycin with Polymixin with Hydrocortisone', 'Tobramycin with Dexamethasone', 'Betamethasone', 'Dexamethasone', 'Prednisolone', 'Almond oil', 'Sodium bicarbonate', 'ICHTHAMMOL IN GLYCERINE PACK','Amoxicillin', 'Azithromycin',  'Ceftazidime','Diazepam', 'Prochlorperazine', 'Betahistine', 'Cinnarizine', 'Dexamethasone', 'Methylprednisolone', 'Prednisolone','Mupirocin', 'Ipratopium bromide', 'Chromic acid', 'Silver nitrate', 'Lidocaine', 'Sodium cromoglycate', 'Neomycin with Betamethasone', 'Oxymetazoline', 'Xylometazoline', 'Beclomethasone', 'Betamethasone', 'Fluticasone', 'Mometasone', 'Glucose in glycerine', 'Dapsone',  'Ampicillin', 'Co-trimoxazole', 'Cetirizine', 'Desloratadine', 'Ebastine', 'Fexofenadine', 'Levocetirizine', 'Loratadine', 'Amphoterecin B', 'Tranexamic acid', 'Turpentine oil', 'BENZALKONIUM CHLORIDE + CHOLINE SALICYLATE', 'Chlorhexidine', 'CHLORHEXIDINE + LIDOCAINE + METRONIDAZOLE', "Hydrogen Peroxide (H₂O₂)", 'NYSTATIN','Povidone iodine',  'Cefdinir',  'Penicillin V',  'Valacyclovir', 'Acetaminophen (Paracetamol)', 'Dexamethasone', 'Pantoprazole', 'Prednisolone']


extract_medicine_data(pdf_file, output_json_file, pages_to_extract)


print(f"✅ Extraction complete! Data saved to {output_json_file}")
print("Initial medicines number = ", len(medicine_list))


output_json = "god_output.json"  # Replace with your actual JSON file name


# Open and load the JSON file
with open(output_json, "r", encoding="utf-8") as file:
   data = json.load(file)


# Count the number of dictionaries (medicine entries)
print(f"✅ Total medicine entries: {len(data)}")


