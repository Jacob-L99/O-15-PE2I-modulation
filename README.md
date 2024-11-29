O-15 Vatten:

Programmet läser in en (eller två) dicom mappar eller .nii filer 

.inf filer väljs för de mappar eller .nii filer som har valts (måste ha lika många frames) 

Klicka på 'Fortsätt' och programmet börjar med att läsa in all data och nerskala till 128x128 bilder.

Registrering skrev med 'ANTspy' (SyN, Symmetric normalization)

Användaren kan sedan kolla igenom alla slices om registreringen blev bra och välja 'ja' eller 'nej'

Vid 'Nej' görs registreringen om, (radom_seed ändras med +2 för varje gång man trycker på 'Nej')

Vid 'Ja' går programmet vidare till beräkningar och visualiseringar 

När allt är klart öppnas en ny ruta med alla bilder, här kan användaren välja vilka slices som syns, samt ända färgskalan (max)

'Spara som .nii' skapar .nii filer i MNI-space och i PAT-space med k_1 (Perfusion) och k_2 (flow-out rate), sparas där programmet finns

"Spara som dicom" fungerar inte just nu, om den knappen trycks, stängs programmet ner

"Skapa PDF" skapar en PDF med all alla bilder (i MNI-space och SSP) samt tabeller med z-scores och medlevärden och hur rörelsekorrigeringen har förflyttat varje frame

PE2I:

Programmet läser in en  dicom mapp 

Klicka på 'Fortsätt' och programmet börjar med att läsa in all data och nerskala till 128x128 bilder.

Registrering skrev med 'ANTspy' (SyN, Symmetric normalization)

Användaren kan sedan kolla igenom alla slices om registreringen blev bra och välja 'ja' eller 'nej'

Vid 'Nej' görs registreringen om, (radom_seed ändras med +2 för varje gång man trycker på 'Nej')

Vid 'Ja' går programmet vidare till beräkningar och visualiseringar 

När allt är klart öppnas en ny ruta med alla bilder, här kan användaren välja vilka slices som syns, samt ända färgskalorna (max)

'Spara som .nii' skapar .nii filer i MNI-space och i PAT-space med R_I och BP, sparas där programmet finns

"Spara som dicom" fungerar inte just nu, om den knappen trycks, stängs programmet ner

"Skapa PDF" skapar en PDF med all alla bilder (i MNI-space och SSP) samt tabeller med z-scores och medlevärden och hur rörelsekorrigeringen har förflyttat varje frame
