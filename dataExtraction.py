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
        "Drug and food interaction": ["Drug and food interaction", "Drug and food interactions", "Interactions", "Interaction"]
   }
  
   def match_attribute(line):
       for key, variants in attribute_variants.items():
           if any(line.lower().startswith(variant.lower()) for variant in variants):
               return key
       return None
   medicine_pattern = re.compile(r"^([A-Z\s\(\)/-]+)(?:\s*\(([^)]+)\))?$")
#    medicine_pattern = re.compile(r"^([A-Z\s\(\)-]+)(?:\s*\(([^)]+)\))?$")
  
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
pages_to_extract = list(range(4, len(doc) + 1))


medicine_list = ['Aluminum hydroxide', 'Magnesium hydroxide', 'Atropine sulphate', 'Drotaverine', 'Flavoxate hydrochloride', 'Hyoscine butyl bromide', 'Mebeverine hydrochloride', 'Cimetidine', 'Esomeprazole', 'Famotidine', 'Lansoprazole', 'Omeprazole', 'Pantoprazole', 'Rabeprazole', 'Ranitidine', 'Sucralfate', 'Cyclizine', 'Dimenhydrinate', 'Domperidone', 'Metoclopramide', 'Ondansetron', 'Prochlorperazine', 'Promethazine', 'Diphenoxylate', 'Loperamide', 'Oral rehydration salts', 'Rifaximin', 'Zinc sulfate', 'Bran', 'Ispaghula husk', 'Docusate sodium', 'Liquid paraffin', 'Drugs used in Gastrointestinal Disorders', 'Bisacodyl', 'Senna', 'Lactulose', 'Magnesium sulphate', 'Adalimumab', 'Budesonide', 'Hydrocortisone acetate', 'Infliximab', 'Mesalazine', 'Sulfasalazine', 'Octreotide', 'Terlipressin', 'Ursodeoxycholic acid', 'Glyceryl trinatre', 'Isosorbide dinitrate', 'Isosobide mononitrate', 'Atenolol', 'Bisoprolol', 'Carvedilol', 'Metoprolol', 'Propranolol', 'Diltiazem', 'Verapamil', 'Adenosine', 'Amiodarone', 'Atenolol', 'Digoxin', 'Disopyramide', 'Isoprenaline', 'Lignocaine', 'Metoprolol', 'Procainamide', 'Propranolol', 'Verapamil', 'Captopril', 'Enalapril', 'Lisinopril', 'Ramipril', 'Irbesartan', 'Losartan', 'Telmisartan', 'Valsartan', 'Atenolol', 'Bisoprolol', 'Metoprolol', 'Labetalol', 'Nebivolol', 'Drugs used in Cardiovascular Disorders', 'Propranolol', 'Amlodipine', 'Felodipine', 'Nifedipine', 'Clonidine', 'Chlorthalidone', 'Hydrochlorothiazide', 'Indapamide', 'Metolazone', 'Spironolactone', 'Torsemide', 'Hydralazine hydrochloride', 'Methyldopa', 'Prazosin', 'Reserpine', 'Sodium nitroprusside', 'Tamsulosin', 'Terazosin', 'Dopamine hydrochloride', 'Dobutamine', 'Phenylephrine', 'Vasopressin', 'Bisoprolol', 'Carvedilol', 'Dobutamine', 'Digoxin', 'Dopamine', 'Furosemide', 'Losartan', 'Metoprolol', 'Milrinone', 'Ramipril', 'Spironolactone', 'Valsartan', 'Bosentan', 'Sildenafil', 'Atorvastatin', 'Cholestyramine', 'Clofibrates', 'Ezetimide', 'Fenofibrate', 'Gemfibrozil', 'Nicotinic acid', 'Simvastatin', 'Rosuvastatin', 'Drugs used in Cardiovascular Disorders', 'Ferrous fumarate', 'Ferrous fumarate with folic acid', 'Ferrous gluconate', 'Ferrous sulphate', 'Ferrous sulfate with ascorbic acid', 'Ferrous sulfate with folic acid', 'Iron dextran', 'Folic acid', 'Epoetin alfa', 'Low molecular weight heparins', 'Dalteparin', 'Enoxaparin', 'Warfarin', 'Acenocoumarol', 'Others', 'Apixaban', 'Bivalirudin', 'Dabigatran', 'Fondaparinux', 'Rivaroxaban', 'Aspirin', 'Clopidogrel', 'Prasugrel', 'Ticlopidine', 'Alteplase', 'Streptokinase', 'Tenecteplase', 'Urokinase', 'Ethamsylate', 'Tranexamic acid', 'Protamine sulfate', 'Drugs used in Blood Disorders', 'Human albumin', 'Factor IX complex', 'Desmopressin', 'Polygeline', 'Calcium', 'Iodine', 'Iron', 'Phosphates', 'Sodium flouride', 'Zinc', 'Vitamin K', 'Acetazolamide', 'Amiloride hydrochloride', 'Chlorthalidone', 'Eplerenone', 'Furosemide', 'Hydrochlorothiazide', 'Indapamide', 'Mannitol', 'Metolazone', 'Spironolactone', 'Torsemide', 'Triamterene', 'Desmopressin', 'Vasopressin', 'Nitrofurantoin', 'Bethanchol', 'Flavoxate', 'Mirabegron', 'Oxybutynin', 'Solifenacin', 'Tolteradin', 'Albumin', 'Glucose', 'Glucose with sodium chloride', 'Potassium chloride', 'Sodium bicarbonate', 'Sodium chloride', 'Drugs used in Renal Disorders', 'Ipratropium bromide', 'Bambuterol', 'Formoterol', 'Salbutamol', 'Salmeterol', 'Terbutaline sulphate', 'Beclomethasone dipropionate', 'Budesonide', 'Hydrocortisone', 'Montelukast', 'Zafirlukast', 'Sodium cromoglycate', 'Ephedrine hydrochloride', 'Epinephrine', 'Aminophylline', 'Doxofylline', 'Theophylline', 'Codeine phosphate', 'Dextromethorphan', 'Noscapine', 'Pholcodine', 'Bromhexine', 'Carbocysteine', 'Acetylcysteine', 'Phenylephrine', 'Pseudoephedrine', 'Caffeine citrate', 'Drugs used in Respiratory Disorders', 'Carbamazepine', 'Clobazam', 'Clonazepam', 'Diazepam', 'Gabapentin', 'Lacosamide', 'Lamotrigine', 'Levetiracetam', 'Oxcarbazepine', 'Phenobarbital', 'Phenytoin', 'Topiramate', 'Valproic acid', 'Amantadine', 'Benztropine', 'Bromocriptine', 'Entacapone', 'Levodopa and carbidopa', 'Oxphenadrine hydrochloride', 'Pramipexole', 'Rasagiline', 'Ropinirole', 'Selegiline hydrochloride', 'Ergotamine tartarate', 'NSAIDS', 'Opioids', 'Rizatriptan', 'Sumatriptan', 'Amitriptyline', 'Flunarizine', 'Lamotrigine', 'Metoprolol', 'Propranolol', 'Sodium valproate', 'Topiramate', 'Psychiatric Disorders', 'Amitriptyline', 'Carbamazepine', 'Duloxetine', 'Gabapentine', 'Lamotrigine', 'Oxcarbazepine', 'Pregabalin', 'Topiramate', 'Acetazolamide', 'Mannitol', 'Buprenorphine', 'Codeine phosphate', 'Methadone hydrochloride', 'Morphine sulphate', 'Tramadol hydrochloride', 'Naloxone', 'Naltrexone', 'Pentazocine', 'Mirtazapine', 'Bupropion', 'Escitalopram', 'Fluoxetine', 'Fluvoxamine', 'Paroxetine', 'Sertraline', 'Trazodone hydrochloride', 'Duloxetine', 'Venlafaxine', 'Amitriptyline', 'Clomipramine', 'Imipramine', 'Nortryptyline', 'Chlorpromazine', 'Fluphenazine decanoate', 'Haloperidol', 'Trifluoperazine', 'Amisulpride', 'Aripiprazole', 'Clozapine', 'Olanzapine', 'Quetiapine', 'Risperidone', 'Atomoxetine', 'Bupropion', 'Donepezil hydrochloride', 'Memantine hydrochloride', 'Rivastigmine', 'Chlordiazepoxide', 'Diazepam', 'Lorazepam', 'Disulfiram', 'Naltrexone', 'Topiramate', 'Naltrexone', 'Codeine', 'Clonidine', 'Buprenorphine', 'Methadone', 'Bupropion', 'Carbamazepine', 'Gabapentine', 'Lamotrigine', 'Lithium', 'Sodium valproate', 'Alprazolam', 'Chlordiazepoxide', 'Clobazam', 'Clonazepam', 'Diazepam', 'Lorazepam', 'Zolpidem', 'IMIPRAMINE', 'Allopurinol', 'Colchicine', 'Febuxostat', 'Probenecid', 'Sulfinpyrazone', 'Abatacept', 'Adalimumab', 'Anakinra', 'Etanercept', 'Infliximab', 'Leflunomide', 'Methotrexate', 'Penicillamine', 'Rituximab', 'Sulfasalazine', 'Tocilizumab', 'Tofacutinib', 'Aceclofenac', 'Aspirin', 'Diclofenac', 'Flurbiprofen', 'Ibuprofen', 'Indomethacin', 'Mefenamic acid', 'Naproxen', 'Paracetamol', 'Piroxicam', 'Celecoxib', 'Etoricoxib', 'Diacerein', 'Alendronate', 'Joint Disorders', 'Etidronate', 'Ibandronate', 'Pamidronate', 'Risedronate', 'Zoledronate', 'Baclofen', 'Dexmedetomidine', 'Midazolam', 'Tizanidine', 'Denosumab', 'Diloxanide furoate', 'Metronidazole', 'Secnidazole', 'Tinidazole', 'Cefaclor', 'Cefadroxil', 'Cefalexin', 'Cefazolin', 'Cefepime', 'Cefixime', 'Cefoperazone', 'Cefotaxime', 'Cefpodoxime', 'Ceftazidime', 'Ceftriaxone', 'Cefuroxime', 'Amoxicillin', 'Ampicillin', 'Bacampicillin', 'Carbenicillin', 'Cloxacillin', 'Flucloxacillin', 'Piperacillin', 'Amikacin', 'Gentamicin', 'Neomycin', 'Clindamycin', 'Linezolid', 'Drugs used in Infections', 'Azithromycin', 'Clarithromycin', 'Erythromycin', 'Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin', 'Nalidixic acid', 'Norfloxacin', 'Ofloxacin', 'trimethoprim', 'Doxycycline', 'Minocycline', 'Oxytetracycline', 'Tetracycline', 'Vancomycin', 'Amphotericin B', 'Clotrimazole', 'Fluconazole', 'Griseofulvin', 'Itraconazole', 'Ketoconazole', 'Nystatin', 'Albendazole', 'Mebendazole', 'Miltefosine', 'Niclosamide', 'Pentamidine', 'Piperazine', 'Praziquantel', 'Pyrantel pamoate', 'Sodium stibogluconate', 'Clofazimine', 'Dapsone', 'Rifampicin', 'Artemether with lumefantrine', 'Artesunate', 'Chloroquine', 'Mefloquine', 'Primaquine', 'Quinine', 'Sulfadoxine and pyrimethamine', 'Ethambutol', 'Isoniazid', 'Pyrazinamide', 'Rifampin', 'Streptomycin', 'Bedaquiline', 'Capreomycin', 'Cycloserine', 'Ethionamide', 'Kanamycin', 'Entecavir', 'Peginterferon alfa', 'Sofosbuvir', 'Acyclovir', 'Oseltamivir', 'Didanosine', 'Efavirenz', 'Indinavir', 'Lamivudine', 'Nelfinavir', 'Nevirapine', 'Ritonavir', 'Saquinavir', 'Stavudine', 'Tenofovir disoproxil', 'Zidovudine', 'Drugs used in Infections', 'Beclomethasone', 'Betamethasone', 'Cortisone acetate', 'Dexamethasone', 'Fludrocortisone acetate', 'Hydrocortisone', 'Methyl prednisolone', 'Prednisolone', 'Triamcinolone acetonide', 'Insulin aspart', 'Insulin glargine', 'Insulin lispro', 'Insulin protamine zinc', 'Insulin soluble', 'Acarbose', 'Metformin', 'Sitagliptin', 'Liraglutide', 'Repaglinide', 'Chlorpropamide', 'Glibenclamide', 'Gliclazide', 'Glipizide', 'Pioglitazone', 'Rosiglitazone', 'Glucagon', 'Levothyroxine', 'Drugs used in Endocrine Disorders', 'Carbimazole', 'Iodine', 'Propanolol', 'Propylthiouracil', 'Octreotide', 'Somatotropin', 'Calcitonin', 'Teriparatide', 'Orlistat', 'Mesterolone', 'Methyl testosterone', 'Testosterone', 'Nandrolone', 'Oxymetholone', 'Stanozolol', 'Bicalutamide', 'Danazol', 'Finasetride', 'Conjugated estrogen', 'Estradiol', 'Ethinylestradiol', 'Mestranol', 'Chorionic gonadotrophin', 'Clomiphene', 'Tamoxifen', 'Dydrogesterone', 'Hydroxyprogesterone', 'Medroxyprogesterone', 'Norethisterone', 'Combined oral contraceptive pills', 'Medroxyprogesterone', 'Mifepristone', 'Oxytocin', 'Drugs used in Reproductive Disorders', 'Nifedipine', 'Isoxsuprine', 'Salbutamol', 'Terbutaline', 'of drugs', 'Aluminium phosphide poisoning', 'Atropine and belladonna poisoning', 'Barbiturate poisoning', 'Benzodiazepine poisoning', 'Carbamate poisoning', 'Carbon monoxide poisoning', 'Chlorinated hydrocarbon insecticide poisoning', 'Corrosives', 'Ethylene glycol poisoning', 'Insect stings', 'Kerosene poisoning', 'Methanol poisoning', 'Morphine and other opioids poisoning', 'Mushroom poisoning', 'Organophosphorus insecticides poisoning', 'Paracetamol poisoning', 'Tricyclic antidepressants poisoning', 'Zinc phosphide poisoning', 'other substances used in Poisoning', 'Actinomycin D', 'Arsenic trioxide', 'Azathioprine', 'Capecitabine', 'Carboplatin', 'Chlorambucil', 'Cisplatin', 'Cyclophosphamide', 'Cytarabine', 'Daunorubicin', 'Docetaxel', 'Doxorubicin', 'Epirubicin', 'Etoposide', 'Fludarabine', 'Fluorouracil', 'Hydroxyurea', 'Irinotecan', 'Interferon beta', 'Lomustine', 'Melphalan', 'Methotrexate', 'Mitomycin', 'Mitoxantrone', 'Oxaliplatin', 'Paclitaxel', 'Pemetrexed', 'Procarbazine', 'Tacrolimus', 'Temozolomide', 'Topotecan', 'Tretinoin', 'Vincristine', 'Vinblastine', 'Vinorelbine', 'Bevacizumab', 'Bortezomib', 'Cetuximab', 'Erlotinib', 'Drugs used in Malignant Diseases', 'Geftinib', 'Imatinib', 'Nilotinib', 'Osimertinib', 'Rituximab', 'Sunitinib', 'Trastuzumab', 'Anastrozole', 'Exemestane', 'Fulvestrant', 'Tamoxifen', 'Antirabies hyperimmune serum', 'Polyvenum antisnake serum', 'Measles vaccine', 'Normal immunoglobulin', 'Rotavirus vaccine', 'Tetanus toxoid', 'Diphtheria antitoxins', 'Hepatitis B immunoglobulin', 'Human papillomavirus vaccine', 'Influenza vaccine', 'Rabies vaccine', 'Typhoid vaccine', 'Yellow fever vaccine', 'Immunologicals', 'Benzoic acid and Salicylic acid', 'Fusidic Acid', 'Metronidazole', 'Mupirocin', 'Silver Sulfadiazine', 'Amorolfine', 'Amphotericin B', 'Clotrimazole', 'Gentian violet', 'Griseofulvin', 'Ketoconazole', 'Miconazole nitrate', 'Neomycin', 'Nystatin', 'Sertaconazole', 'Terbinafine', 'Benzyl Benzoate', 'Ivermectin', 'Permethrin', 'Aciclovir', 'Clobetasone butyrate', 'Clobetasol propionate', 'Flucinolone acetonide', 'Fluticasone', 'Hydrocortisone', 'Hydrocortisone butyrate', 'Tacrolimus', 'Triamcinolone', 'Acitretin', 'Adapalene', 'Adapalene with Benzoyl peroxide', 'Isotretinoin', 'Tazarotene', 'Drugs used in Skin Disorders', 'Coal tar', 'Coal tar with Salicylic acid and Precipitated Sulfur', 'Calcipotriol', 'Calcitriol', 'Aluminium chloride hexahydrate', 'Fluorouracil', 'Doxepin', 'Clindamycin', 'Azelaic acid', 'Benzoyl peroxide with Clindamycin', 'Erythromycin', 'Selenium sulphide', 'Minoxidil', 'Eflornithine', 'Chlorhexidine', 'Imiquimod', 'Potassium permanganate', 'Salicylic acid', 'Azelaic acid', 'Capsaicin', 'Hydroquinone', 'Psoralen', 'Chloramphenicol', 'Ciprofloxacin', 'Gentamicin', 'Levofloxacin', 'Moxifloxacin', 'Neomycin', 'Ofloxacin', 'Polymyxin B', 'Tetracycline', 'Acyclovir', 'Ganciclovir', 'Idoxuridine', 'Fluconazole', 'Natamycin', 'Betamethasone', 'Dexamethasone', 'Diclofenac', 'Fluorometholone', 'Flurbiprofen', 'Hydrocortisone', 'Indomethacin', 'Ketorolac', 'Prednisolone', 'Ketotifen', 'Olopatadine', 'Sodium chromoglycate', 'Acetazolamide', 'Bimatoprost', 'Brimonidine', 'Dorzolamide', 'Latanoprost', 'Mannitol', 'Pilocarpine', 'Timolol', 'Drugs used in Ophthalmic Disorders', 'Travoprost', 'Atropine', 'Cyclopentolate', 'Homatropine', 'Phenylephrine', 'Tropicamide', 'Acetylcysteine', 'Bevacizumab', 'Bupivacaine', 'Carbomer', 'Cyclosporine', 'Fluorescein sodium', 'Hypromellose', 'Liquid paraffin', 'Polyvinyl alcohol', 'Sodium chloride', 'Triamcinolone', 'Chloramphenicol', 'Ciprofloxacin', 'Gentamicin', 'Ofloxacin', 'Clotrimazole', 'Chloramphenicol with Dexamethasone', 'Ciprofloxacin with Hydrocortisone', 'Neomycin with Polymixin with Hydrocortisone', 'Tobramycin with Dexamethasone', 'Betamethasone', 'Dexamethasone', 'Prednisolone', 'Almond oil', 'Sodium bicarbonate', 'Amoxicillin', 'Azithromycin', 'Cefixime', 'Ceftazidime', 'Ceftriaxone', 'Chloramphenicol', 'Ciprofloxacin', 'Clarithromycin', 'Flucloxacillin', 'Gentamicin', 'Diazepam', 'Prochlorperazine', 'Betahistine', 'Cinnarizine', 'Prednisolone', 'Dexamethasone', 'Methylprednisolone', 'Mupirocin', 'Ipratopium bromide', 'Chromic acid', 'Silver nitrate', 'Lidocaine', 'Sodium cromoglycate', 'Neomycin with Betamethasone', 'Oxymetazoline', 'Xylometazoline', 'Beclomethasone', 'Betamethasone', 'Fluticasone', 'Mometasone', 'Glucose in glycerine', 'Dapsone', 'Doxycycline', 'Flucloxacillin', 'Rifampicin', 'Ampicillin', 'Co-trimoxazole', 'Cetirizine', 'Desloratadine', 'Ebastine', 'Fexofenadine', 'Levocetirizine', 'Loratadine', 'Amphoterecin B', 'Tranexamic acid', 'Turpentine oil', 'Amphotericin B', 'Chlorhexidine', 'Clotrimazole', 'Fluticasone', 'Hydrocortisone', 'Hydrogen Peroxide', 'Lidocaine', 'Mometasone', 'Povidone iodine', 'Amoxicillin', 'Azithromycin', 'Benzyl penicillin', 'Cefdinir', 'Cefuroxime axetil', 'Ceftriaxone', 'Clarithromycin', 'Clindamycin', 'Diptheria antitoxin', 'Erythromycin', 'Levofloxacin', 'Penicillin V', 'Acyclovir', 'Valacyclovir', 'Fluconazole', 'Itraconazole', 'Ketoconazole', 'Acetaminophen', 'Dexamethasone', 'Pantoprazole', 'Prednisolone']


extract_medicine_data(pdf_file, output_json_file, pages_to_extract)


print(f"✅ Extraction complete! Data saved to {output_json_file}")
print("Initial medicines number = ", len(medicine_list))


output_json = "god_output.json"  # Replace with your actual JSON file name


# Open and load the JSON file
with open(output_json, "r", encoding="utf-8") as file:
   data = json.load(file)


# Count the number of dictionaries (medicine entries)
print(f"✅ Total medicine entries: {len(data)}")


