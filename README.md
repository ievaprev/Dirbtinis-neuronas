# Dirbtinis_neuronas


**Darbo tikslas:** Išanalizuoti dirbtinio neurono modelio veikimo principus.

**Darbo uždaviniai:**

* Sugeneruoti du dvimačių duomenų rinkinius, kurie būtų sudaryti iš 10 duomenų įrašų.
* Suteikti galimybę vartotojui pasirinkti aktyvacijos funkciją.
* Rankiniu būdu apskaičiuoti klasifikavimui tinkančius svorius ir poslinkį.
* Sukurti dirbtinį neuroną, kuris naudodamas išrinktus svorius ir poslinkius gebėtų klasifikuoti duomenis.
* Pavaizduoti atskiras klases ir jas skiriančias tieses grafike.

### 1. Duomenų generavimas
- Sugeneruojamos dvi duomenų grupės (c1 ir c2), kiekvienoje po 10 dvimačių taškų, kurios yra tiesiškai atskiriamos.

### 2. Aktyvacijos funkcijos pasirinkimas
- Vartotojas renkasi aktyvacijos funkciją: slenkstinę (step_function) arba sigmoidinę (torch.sigmoid).

### 3. Dirbtinio neurono kūrimas
- Sukuriama neurono klasė naudojant PyTorch, su galimybe pritaikyti pasirinktą aktyvacijos funkciją.

### 4. Tinkamų svorių ir poslinkio paieška
- Naudojant atsitiktinį paieškos algoritmą, kodas ieško 3 svorių (w) ir poslinkio (b) rinkinių, kurie teisingai klasifikuoja visas duomenų įrašų klases.

### 5. Klasifikavimas ir vizualizacija
- Dirbtinis neuronas su nustatytais svoriais ir poslinkiu testuojamas su duomenimis.
- Klasių priklausomybė pateikiama grafiškai: skirtingos spalvos klasėms, tiesės – klasėms atskirti.
- Tiesės brėžiamos pagal skirtingus surastus svorių ir poslinkio rinkinius.

## Kodas – pagrindiniai žingsniai

- `Neuron` – neurono klasė, kuri paveldi iš `nn.Module`, su pasirenkama aktyvacijos funkcija.
- `step_function` – slenkstinė aktyvacijos funkcija (grąžina 1, kai įėjimas >=0).
- Duomenų generavimas – naudotas `numpy` ir `numpy.random.seed`, užtikrinant, kad duomenys bus tapatūs kiekvieną kartą paleidus kodą.
- Svorių ir poslinkio paieška – 1000 bandymų generuoti atsitiktinius svorius bei poslinkį, ieškant rinkinių, kurie visiškai tiksliai klasifikuoja duomenis.
- Rezultatų vizualizacija – panaudojant `matplotlib` pavaizduojamos klasės ir jas atskiriančios tiesės.

## Reziumė

Šis darbas skirtas supažindinti su paprasto dirbtinio neurono modelio veikimu ir duomenų klasifikavimu naudojant ranka parinktus parametrus. Įgyvendinamas visas procesas nuo duomenų generavimo iki vizualizavimo ir paremtas interaktyviu vartotojo pasirinkimu.
