const regex = /'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu;



class BPETokenizer {
    constructor(neuralNetwork) {
        this.vocab = new Set();
        this.tokenToID = new Map();
        this.maxSubwordLength = 5;
        this.subwordFrequencies = new Map();
        this.neuralNetwork = neuralNetwork;
    }

    learnVocab(corpus, numIterations) {
        for (const word of corpus) {
            this.updateVocab(word);
        }

        for (let iter = 0; iter < numIterations; iter++) {
            const pairs = this.getSubwordPairs();
            if (pairs.size === 0) break;

            const mostCommonPair = this.findMostCommonPair(pairs);
            if (!mostCommonPair) break;

            this.vocab.add(mostCommonPair);
            corpus = corpus.map(word => word.replace(new RegExp(mostCommonPair, 'g'), ''));
        }

        this.cacheSubwordFrequencies();
        this.buildTokenToID();
    }

    getSubwordPairs() {
        const pairs = new Map();
        for (const word of this.vocab) {
            for (let len = 2; len <= this.maxSubwordLength; len++) {
                for (let i = 0; i < word.length - len + 1; i++) {
                    const subword = word.slice(i, i + len);
                    pairs.set(subword, (pairs.get(subword) || 0) + 1);
                }
            }
        }
        return pairs;
    }

    findMostCommonPair(pairs) {
        let maxCount = -1;
        let mostCommonPair = null;
        for (const [pair, count] of pairs.entries()) {
            if (count > maxCount) {
                mostCommonPair = pair;
                maxCount = count;
            }
        }
        return mostCommonPair;
    }

    updateVocab(word) {
        for (let len = 1; len <= this.maxSubwordLength; len++) {
            for (let i = 0; i < word.length - len + 1; i++) {
                const subword = word.slice(i, i + len);
                this.vocab.add(subword);
            }
        }
    }

    cacheSubwordFrequencies() {
        for (const word of this.vocab) {
            for (let len = 1; len <= this.maxSubwordLength; len++) {
                for (let i = 0; i < word.length - len + 1; i++) {
                    const subword = word.slice(i, i + len);
                    this.subwordFrequencies.set(subword, (this.subwordFrequencies.get(subword) || 0) + 1);
                }
            }
        }
    }

    buildTokenToID() {
        let nextID = 0;
        for (const token of this.vocab) {
            this.tokenToID.set(token, nextID++);
        }
    }
    purgeVocabulary(targetVocabSize) {
        // Convert the vocabulary Set to an array for sorting
        const sortedVocab = Array.from(this.vocab);
    
        // Sort the vocabulary by frequency and length combined
        sortedVocab.sort((a, b) => {
            // Compare frequencies
            const freqDiff = this.subwordFrequencies.get(b) - this.subwordFrequencies.get(a);
            if (freqDiff !== 0) {
                return freqDiff;
            }
            // If frequencies are equal, prioritize longer tokens
            return b.length - a.length;
        });
    
        // Keep only the top targetVocabSize tokens
        const purgedVocab = sortedVocab.slice(0, targetVocabSize);
    
        // Rebuild the tokenToID map
        const tokenToID = new Map();
        for (let i = 0; i < purgedVocab.length; i++) {
            tokenToID.set(purgedVocab[i], i);
        }
    
        // Update the vocabulary and tokenToID
        this.vocab = new Set(purgedVocab);
        this.tokenToID = tokenToID;
    
        // Log the size of the vocabulary
        console.log("BPE Vocabulary Size:", this.vocab.size);
    
        // Update nextID
        this.nextID = purgedVocab.length;
    }
    
    
    isRelativelyUnique(token, uniqueThreshold) {
        // Calculate the number of other tokens that contain the given token as a substring
        const numContainingTokens = Array.from(this.subwordFrequencies.keys()).filter(t => t.includes(token)).length;
    
        // Calculate the uniqueness ratio
        const uniquenessRatio = numContainingTokens / this.vocab.size;
    
        // Return true if the uniqueness ratio is above the threshold
        return uniquenessRatio < uniqueThreshold;
    }
    
    
    encode(text) {
        const { tokens, tokenIDs } = this.tokenize(text);
        return tokenIDs;
    }

    decode(tokenIDs) {
        let decodedText = '';
        for (const tokenID of tokenIDs) {
            const token = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === tokenID);
            decodedText += token;
        }
        return decodedText;
    }
    
tokenize(text) {
    const tokens = [];
    const tokenIDs = [];
    
  
    const words = text.match(regex)

    for (const word of words) {
        let tokenizedWord = '';
        let i = 0;
        while (i < word.length) {
            let found = false;
            for (let len = this.maxSubwordLength; len >= 1; len--) {
                if (i + len <= word.length) {
                    const subword = word.slice(i, i + len);
                    if (this.vocab.has(subword)) {
                        tokenizedWord += subword + ' ';
                        tokenIDs.push(this.tokenToID.get(subword));
                        i += len;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                tokenizedWord += word[i];
                tokenIDs.push(this.tokenToID.get(word[i]));
                tokenizedWord += '';
                i++;
            }
        }
        // Trim and split the tokenized word to remove extra spaces
        tokens.push(...tokenizedWord.trim().split(/\s+/));
    }
    return { tokens, tokenIDs };
}


    clusterTokens(tokens, similarityThreshold) {
        function cosineSimilarity(vec1, vec2) {
            let dotProduct = 0, magnitude1 = 0, magnitude2 = 0;
            for (let i = 0; i < vec1.length; i++) {
                dotProduct += vec1[i] * vec2[i];
                magnitude1 += vec1[i] ** 2;
                magnitude2 += vec2[i] ** 2;
            }
            magnitude1 = Math.sqrt(magnitude1);
            magnitude2 = Math.sqrt(magnitude2);
            return dotProduct / (magnitude1 * magnitude2);
        }

        function hierarchicalClustering(tokens, similarityThreshold) {
            let clusters = tokens.map(token => [token]);
            let similarities = Array.from({ length: tokens.length }, () => Array(tokens.length).fill(0));

            for (let i = 0; i < tokens.length; i++) {
                for (let j = 0; j < tokens.length; j++) {
                    similarities[i][j] = cosineSimilarity(tokens[i], tokens[j]);
                }
            }

            while (clusters.length > 1) {
                let maxSimilarity = -1;
                let mostSimilarPair = [];

                for (let i = 0; i < clusters.length; i++) {
                    for (let j = i + 1; j < clusters.length; j++) {
                        let clusterSimilarity = 0;
                        for (let token1 of clusters[i]) {
                            for (let token2 of clusters[j]) {
                                clusterSimilarity += similarities[token1][token2];
                            }
                        }
                        clusterSimilarity /= clusters[i].length * clusters[j].length;

                        if (clusterSimilarity > maxSimilarity) {
                            maxSimilarity = clusterSimilarity;
                            mostSimilarPair = [i, j];
                        }
                    }
                }

                clusters[mostSimilarPair[0]] = clusters[mostSimilarPair[0]].concat(clusters[mostSimilarPair[1]]);
                clusters.splice(mostSimilarPair[1], 1);
                similarities.splice(mostSimilarPair[1], 1);
                similarities.forEach(row => row.splice(mostSimilarPair[1], 1));
            }

            clusters = clusters.filter(cluster => cluster.length > 1 || similarityThreshold <= 1);
            return clusters;
        }

        const tokenVectors = tokens.map(token => this.neuralNetwork.charToInput(token));
        return hierarchicalClustering(tokenVectors, similarityThreshold);
    }

   
}

// Example usage:
const corpusText = `V hustom lese, by ste okrem všelijakých zvieratiek natrafili i na skromného medvedíka menom Brumík. Hnedý medvedík bol známy po celom šírom lese svojou milou a dobrou povahou. Všetkých mal rád a bol to ten najpriateľskejší medvedík, akého ste kedy streli.

Bol síce priateľský no svojou veľkosťou pôsobil strašidelne. Veľa zvieratiek sa ho preto bálo. Vždy keď sa túlal po lese a hľadal sladký medík či zbieral všelijaké chutné bobuľky, pozdravil každé stvorenie na ktoré natrafil. Nie všetci ho však odzdravili.  Chladné zimné večery prečkával vo svojom brlohu, kde hlasne vyfukoval zo spánku. Niektoré zvieratká, ako medvede sa totiž cez zimu poberajú na dlhý spánok, aby načerpali sily na dlhé letné potulky lesom. Pred svojím brlôžkom mal Brumík malý potôčik, kde sa okrem umývania rád venoval i chytaniu rýb. Tie miloval zo všetkého najviac.

Ako každé ráno sa z východom slniečka vybral medvedík na  prechádzku. Počasie bolo chladnejšie a chladnejšie a medvedík si musí urobiť zásoby na zimu, keď bude schovaný v brlohu. Vykukol von, zobral si prútený košík a vybral sa na maliny. Dnes musí toho nazbierať za plný košík! Veselo si pospevoval a hádzal do košíka chutné bobuľky. I keď ich viac spapal než do košíka dal.

Rozprávka pre deti - Brumík a vtáčik
Brumík a vtáčik
Ako si tak oberal červenkasté maliny, začul smutný hlások ako kňučí opodiaľ.

„Čo to len môže byť?“ povedal si medvedík Brumík a vybral sa preskúmať tajomný zvuk. Prešiel cez ľadový žblnkajúci potôčik, obišiel vysoký dub a nakukol za malý krík. A čo nevidel!?

Za kríkom ležal na zemi zamotaný vtáčik, ktorý sa snažil dostať z nepoddajného kríku. Metal sa a metal no krídla nie a nie sa pohnúť. V konároch bol zamotaný až-až! Keď zbadal veľkého medveďa začal kričať a kvíliť ešte viac. Bál sa.

Brumík sa len usmial, položil košíček a čupol si k malému vtáčikovi.
Za horami za dolami, kde voda sa sypala a piesok sa lial…schovával sa čarovný les. V tomto lese, by ste natrafili na čary nevídané a zvieratá kúzelné. Všetci si pomáhali a v lese vládol pokoj a radosť. Ľudia však o tomto lese nevedeli. A tak to muselo zostať.

V čarovnom lese žili nie len kúzelné zvieratá, ale i magické bytosti ako víly, čarodejnice či škriatkovia. A všetci spolu žili v harmónií. Zvieratká sa veselo v lese hrali a nažívali si, veselí drobní škriatkovia nosili šťastie všetkým bytostiam v lese. Šťastie nosili práve so svojimi kúzelnými štvorlístkami, ktoré fúkali po okolí. Každý kto zelený štvorlístok našiel, mal veľké šťastie. Kúzelné víly boli nádherné bytosti. 

So svojimi trblietavými krídlami lietali po okolí a starali o to aby les prekvital a aby ani jedno zvieratko či magická bytosť nezostalo hladné. Čarodejnice zase liečili každého kto to potreboval. Používali na to svoje kúzelné elixíry. Biela múdra sova menom Vedma, dávala pozor na celý les, aby ich kúzla a mágie boli v bezpečí. Bola veľmi múdra a každému vedela poradiť.

Rozprávka pre deti - Čarovný les
Keď príde jeseň, vonku často fúka. Vietor sa snaží pomôcť stromom, aby zhodili čo najviac listov. A tak fúka a fúka na všetky strany. Takýto vietor je najlepší na púšťanie šarkanov. Preto sa aj v jednom malom mestečku deti dohodli, že si usporiadajú šarkaniádu drakov. Všetky deti mali po škole prísť na veľkú lúku kde budú púšťať svoje šarkany. To ale ešte nevedeli, kto za nimi na lúku príde.

Veľký deň sa blížil. Všetky deti sa už nevedeli dočkať púšťania papierových drakov. Celé dni doma vyrábali šarkanov. Viazali a lepili špajdle, používali tie najkrajšie farby a viazali na ne tie najdlhšie šnúry. Keď prišlo popoludnie, deti sa zhromaždili na lúke a s pomocou rodičov začali púšťať svoje šarkany.

Rozprávka na dobrú noc - Prekvapenie na šarkaniáde

Ľudmilka bolo malé blonďavé dievčatko, ktoré trávilo väčšinu času v lese. Rada si čítala, odpočívala a dávala srnkám do kŕmidiel rôzne pochúťky. Milovala, keď si mohla len tak ľahnúť do machu, cítiť vôňu hríbov a ochutnávať lesné plody, ktoré rástli všade naokolo. Jedného dňa však Ľudmilka v lese stretla niekoho, koho ešte nikdy predtým nevidela. Ani vlastne netušila, že niekto taký existuje.

Bolo krásne, farebné jesenné popoludnie. Vzduch už bol chladný, ale slnko ešte stále krásne hrialo.Ľudmilka sa práve prechádzala okolo papradia a hľadala nejaké hríby ukryté niekde v zákutí, keď v tú chvíľu počula slabé cupitanie. Otočila sa, ale nikto nikde nebol.
A tak sa z Bobbyho stal právoplatný člen pirátskej posádky. Všetci piráti ho mali veľmi radi a to aj vtedy, keď robil neplechu. Začínali byť však trocha nespokojní. Všetci na lodi niečo robili. Len Bobby nerobil nič. Nemal žiadnu úlohu. Nikto sa však neodvážil nič povedať, pretože sa báli, že sa Jednonôžka nahnevá. Ten mal Bobbyho veľmi rád a rozmaznával ho.
Ďaleko v indiánskej krajine žilo malé dievčatko. Mala dlhé čierne havranie vlasy, vždy spletené do dvoch dlhých vrkočov. Na hlave nosila pestrofarebnú čelenku v ktorej mala napichané pierka. Volala sa Inčučuna. Jej mamička a otecko boli tiež indiáni a žili vo vysokých stanoch.

Inčučuna rada žila ako indiánka. Vedela rozložiť oheň bez zápaliek alebo maľovať na kamene. Ako správna indiánka rozumela aj zvieratám . Keď bola Inčučuna staršia, čakala ju skúška. Každá indiánka mala svojho vlastného koňa. Na ničom inom nejazdili. Kone boli pre nich najlepšími priateľmi a pomocníkmi. Lenže indiáni si kone nekupovali. Museli si nájsť divokého koňa, skrotiť ho a zblížiť sa s ním. A to nebolo ľahké. A práve takáto skúška čakala indiánku Inčučunu. Musela si nájsť vlastného koňa.

Rozprávka pre deti - Príbeh o indiánke
Príbeh o indiánke
Už niekoľko mesiacov chodila na lúku za lesom, kde sa sem-tam objavili divoké kone. Jeden z nich sa jej veľmi páčil. Bol čierny ako jej vlasy. Jeho lesklá hriva sa leskla na slnku. Bol krásny a Inčučuna si ho veľmi chcela skrotiť. Chcelo to však čas a veľa trpezlivosti.
Za deviatimi riekami a deviatimi horami stála malá chalúpka. Vyzerala ako perníková, ale nebola. Bola  úplne celá drevená. Jej okná boli maľované a z komína vždy vychádzal dym z voňavého dreva. V tej chalúpke žil rozprávkový dedko. Bol to najlepší rozprávač rozprávok. Deti a zvieratká z celého okolia chodili k dedkovi počúvať jeho rozprávky.

Jedného dňa sa v blízkosti dreveného domčeka potulovali dvaja chlapci. Nevedeli, čo majú robiť, a tak z nudy rozkopávali huby, lámali kríky a šliapali na kvety. Dedko ich chvíľu pozoroval, potom vyšiel z chalúpky a spýtal sa ich: „Čo to robíte? Prečo takto ničíte všetko okolo seba?“

Obaja chlapci sa zarazili. Nevedeli, že ich niekto vidí. „Dobrý deň, dedko. Nevedeli sme, že tu niekto je. Sme bratia Vilko a Viktor. A veľmi sa nudíme. Nevieme, čo máme robiť,“ priznali sa chlapci.

Rozprávka na čítanie - Rozprávkový dedko
Rozprávkový dedko
„No to ale neznamená, že tu musíte vyvádzať a ničiť, čo nájdete. Ak nemáte čo robiť, sadnite si sem na lavičku pred moju chalúpku a chvíľu počkajte,“ odpovedal dedko a vošiel do domu.

Po chvíli sa objavil vonku s dvoma hrnčekami v ruke. Voňali čerstvo uvareným kakaom. Podal ich chlapcom a povedal: „Budem vám rozprávať príbeh, ktorý sa stal kedysi dávno.“ Vilko a Viktor si každý pevne držali svoje hrnčeky a pomaly popíjali horúci nápoj. Spočiatku sa im nechcelo dedka počúvať. Mysleli si, že to bude nejaká nudná rozprávka, ktorú už aj tak poznali. Netušili však, čo sa bude diať.

Len čo dedko začal rozprávať, zleteli sa okolo nich vtáky, pribehli srnkyi, zajace vyšli z nory a ježkovia tiež rýchlo pricupitali. Všetky zvieratá si sadli tesne k chlapcom a pozorne počúvali. Keď dedko rozprával o jeseni, zrazu začalo padať lístie a jemne sa zdvihol vietor. Keď rozprával o hudbe, vtáky začali krásne štebotať. Keď rozprával o kúzlach, z lesa priletel trblietavý vír a prelietaval medzi chlapcami a zvieratkami.

Vilko a Viktor sa cítili ako v rozprávke. Držali svoje hrnčeky a s otvorenými ústami hltali každé slovo, ktoré dedko povedal. Nevedeli, či sa majú pozerať na zvieratká, alebo na zázraky, ktoré sa diali pri rozprávaní okolo nich. Keď dedko skončil, zvieratká spokojne odcupitali späť do lesa.

„Tak čo, páčil sa vám príbeh, alebo to bola nuda?“ spýtal  sa prekvapených chlapcov dedko. „Bolo to skvelé!“ zvolali obaja naraz. „Keď sa budete opäť nudiť, namiesto ničenia po okolí, príďte zas na kakao a rozprávku. Vždy si na vás nájdem čas,“ usmial sa na nich dedko a rozlúčil sa s nimi.

Odvtedy Vilko s Viktorom chodia do chalúpky pravidelne. Vždy si sadnú na lavičku a volajú: „Dedko, rozprávaj!“ Zistili, že nič iné nepotrebujú. Jediné, čo potrebujú, je čas, ktorý im dedko venuje. Pretože čas od rodičov a starých rodičov je pre deti to najdôležitejšie. Je pre ne čarovný aj bez kúziel.

Koňa sledovala už nejaký čas. Jedného dňa si indiánka sadla priamo na lúku. Ticho tam sedela a čakala. Po chvíli počula dupot kopýt. Položila ruku na zem, zavrela oči a cítila, ako sa zem chveje, keď kone bežia. Celé stádo priklusalo na lúku. Niektoré chrúmali šťavnatú trávu, iné len tak pobehovali, ale najčernejší z nich sa obzeral za Inčučunou. Opatrne sa k nej priblížil a pokúsil sa ju očuchať.

„Len kľud. Musím byť pokojná. Poď sa na mňa pozrieť, fešák, len poď,“ hovorila si indiánka potichu a ani sa nepohla. Nechcela ho vyplašiť. Keď bol kôň takmer pri nej, odfrkol si a odbehol. Inčučuna to ale nevzdala. Vedela, že musí byť trpezlivá. Každý deň chodila na lúku, sedela tam a čakala, kým kone prídu. Potom nechala čierneho koňa, aby ju pozoroval a občas sa jej dotkol.

Celý čas keď sa kôň pokúšal nesmelo priblížiť k Indiánke, sedela trpezlivo a pokojne.

Jedného dňa, keď Inčučuna sedela na lúke a čakala na koňa, z ničoho nič z lesa vybehol vlk. Mladá indiánka úplne stuhla. Keby začala utekať, vlk by ju chytil. A tak ďalej sedela a dúfala, že odíde alebo sa bude brániť. Vlk obchádzal Inčučunu a pozeral na ňu s vycerenými zubami. V tej chvíli Indiánka pocítila, ako zem duní. Kone sa blížili.

Keď sa objavili na lúke a čierny kôň uvidel Inčuunu sedieť uprostred a vlka pri nej, neváhal ani sekundu a rozbehol sa k nej. Postavil sa pred ňu na zadné a začal erdžať. Silno kopal nohami a chránil indiánku svojím telom. Vlk vedel, že kôň bráni Inčučunu. Sklopil uši a odišiel preč.

Potom sa čierny kôň obrátil k mladému dievčaťu, 
Môže si svokra súdom vynútiť stretávanie s vnučkou?
27. mar 2024
Ahojte, prišla som si sem po radu.

S manželom mám 9 ročnú dcéru, ktorá má silnú alergiu na roztoče. V podstate celý chod domácnosti sme prispôsobili jej alergii. Mávala veľmi silné alergické reakcie, niekoľkokrát sme boli na pohotovosti. Alergologička nám odporúčila mimo antihistaminík prať na vysoké teploty, sušičku a všetko čo malá nosí aj žehliť. S manželom žehlíme aj všetko naše oblečenie, obliečky, bytový textil. Je to veľa roboty ale odkedy to robíme, tak sa malej veľmi zlepšil stav.
Ale k problému. Raz za čas si ju zoberie "na stráženie" moje svokra. Je to vždy víkend, v piatok po škole ide tam a v nedeľu príde späť. Dcérka má alergiu zistenú tretí rok. Vždy keď tam dcéra ide prízvukujeme svorke aby jej vyprala povliečky a vyžehlila. Rovnako uteráky. Ona na naše "reči" proste kašle a myslí si, že si to proste vymýšlame, že malej nič nie. Myslela som si, že sa jej to len nechce robiť tak manžel navrhol, že teda keď tam malá pôjde zabalíme jej aj povliečky aj uteráky nech s tým nemá robotu.
Naposledy keď tam bola už to zašlo ďaleko. Manžel bol pre ňu a už keď ju nakladal do auta tak dcéra totálne opuchnutá hlava. Išiel s ňou rovno na pohotovosť, lebo v aute začala hovoriť, že sa jej zle dýcha. 4 dni v nemocnici. Keď sa vrátila domov mali sme seriózny rozhovor s manželom. Ja som sa malej spýtala ako bolo u babky. Malá má s nami veľmi dobrý vzťah a uvedomuje si svoju situáciu. Vyliezlo z nej, že si sama musela obliekať periny, lebo že to je zbytočné. Nezdalo sa mi to ale ja tak. Tak nakoniec mi malá povedala, že jej babka zakázala dať tabletky, že to je len "placebo". Manžel potom volal s mamou, že teda toto neprichádza do úvahy, že už viackrát odignorovala naše pravidlá a teraz doslova vystavila našu dcéru ohrozeniu života a že bez našej prítomnosti tam už dcéra nepôjde a na noc už vôbec nie. Na to sa začala oháňať tým, že ona nás dá na súd, že jej musíme umožniť stretávanie.

Môže niečo také urobiť?

Strana

z5

kvietok2012
•
27. mar 2024
Veď stretávať sa moze, ale iba vo vašej prítomnosti.

iritka
•
27. mar 2024
Sa len vyhraza. Ved stretavanie umoznene ma, spat tam nemusi. Dovod si obhajit viete.

kitycat
•
27. mar 2024
Vsak nech vyrukuje pred sud s tym, ze zanedbava zdravie vnucky 😅

petrao2
•
27. mar 2024
No nech sa paci nech da na sud
Myslim ze kazdy normalny clovek pochopi kde je chyba
Dceru by som tam uz nikdy nedala

oslo1
•
27. mar 2024
Ked na na sud , tak dajte na nu treatne oznamenie za ublizenie na zdravi. Lebo ona to urobila umyselne.
A toto by som dala aj do vyjadrenia, presne datum a cas, keby doslo k sudu.
Moj syn ma alergiu na arasidy, raz dosiel zo skoly s tycinkou Sneakers, lebo spoluziak Mal narodeniny. Si mala vidiet, Ake boli fukoty v skole a co som materi spoluziaka povedala.

ibatanecnica
•
27. mar 2024
môže to urobiť, ale súd, po zvážení dôkazov nemusí jej žiadosti vyhovieť.
@oslo1 ale toto si má ustriehnuť triedna, tá mama spolužiaka nemusela vedieť, že sú tam alergici. Ľuďom nedochádzajú takéto veci, pretože to neriešia, nemajú to v rodine.
A pri väčších deťoch si to deti musia ustrážiť samé. Alebo si myslíš, že sa s orieškami už v živote nestretne?

lucia13022023
•
27. mar 2024
Ja by som tam svoje dieťa bez mojej alebo mužovej prítomnosti nedala, nikdy. Nech vás len dá na súd, ona nie je zákonný Zástupca, nemá právo na vašu dcéru.

oslo1
•
27. mar 2024
@ibatanecnica
V skole maju strictly non peanuts policy. Sme v UK, kde v kazdej jednej tried je dieta alergicke na oriesky.
Triedny sa mi ospravedlnil, ale toto ani do skoly nemalo prist. Nastastie syn je mudry a vzdy sa opytaj, ci ma nieco oriesky.

ibatanecnica
•
27. mar 2024
@lucia13022023 právo na dcéru nemá, ale právo na stretávanie sa áno. Jasné, záleží na situácii

ibatanecnica
•
27. mar 2024
@oslo1 tak to je potom iné. To som nevedela. U nás to funguje úplne inak. Nijako. Buď informuješ učitela alebo dieťa musí vedieť, čo môže a čo nie.

martul
•
27. mar 2024
Ty si tu už raz písala, keď nie tak to bola tvoja dvojníčka.
Môže dať svokra návrh na súd a ten určí stretávanie.

lucia13022023
•
27. mar 2024
@ibatanecnica je na to nejaké súdne právo?

lucia13022023
•
27. mar 2024
@ibatanecnica alebo skôr sa tak spýtam, že môže určiť súd aby sa starý rodič stretával s vnučkou? Dat starej mame nejaký stanovený čas kedy sa môže vidieť s vnučkou?

martul
•
27. mar 2024
@lucia13022023 samozrejme, súd určí stretávanie starej mamy s vnučkou a rodičia sú povinní to dodržiavať. Tak isto ako keď sa manželia rozvedú a súd určí stretávanie.
AUTOR
•
27. mar 2024
@martul Takže sa reálne môže stať že súd povie že tam musí chodiť bez našeho dozoru?

gejb
•
27. mar 2024
@lucia13022023 môže. Ale to by ho nemohla ohrozovať ako táto babka. Tej súd dieťa nedá. Ak maximálne za prítomnosti rodiča. Ale neverím ze bude riskovať také pokazenie vzťahov...mňa keby dá mama na súd v živote s nou už neprehovorim.

oslo1
•
27. mar 2024
@ibatanecnica
Tu maju kazdy jednu alergiu zaznecenu, epi pen plus antihistamines nachystane.
Ja tiez musim mat anaphylaxis kit so mnou, ked chodim po pacientoch.

Ale naspat k autorke. Nedala by som tam dceru. V 21 storoci sa Bude umierat na alergie, nepotrebujes, aby dcera zvysila pocty.

ibatanecnica
•
27. mar 2024
@lucia13022023 áno môže,

matullienka
•
27. mar 2024
Neviem koľko rokov má tvoje dieťa aké ak vy predložíte pred súd dôkazy v akom stave išla vnučka od babičky tak dosť pochybujem, že nejaký sudca, ktorý je pri zmysloch je takéto stretávanie umožní. Druhá vec je, že by som sa spýtala dieťaťa či tam chce chodiť a nech by babke sama povedala, že tam ísť nechce

ibatanecnica
•
27. mar 2024
autorka - nemusí, on môže stanoviť stretávanie sa ale ak budete argumentovať tým, že dieťa je ohrozené po zdravotnej stránke môže nariadiť stretávanie sa mimo domu, v prítomnosti rodiča. a pod.
vonku, u vás atď...

simca06022011
•
27. mar 2024
No v tomto prípade by som so svokrou viac nediskutovala. O návštevách vnučky u nej, prespavani absolútne. Nech jej manžel.oznami ze ak chce vidieť vnučku, môže prísť na návštevu = vybavene.
Nech ide na súd, na lamparen, kamkoľvek... nie je nikde prikazane ze ja ako rodič musím niekomu dieťa poskytovať.
A roztoče... ak nezehli, meprezlieka prádlo atď, roztočov ma plne aj matrace atď.
Minimálne dieťa mobil a kontrolovať branie liekov atď.

lucia13022023
•
27. mar 2024
@gejb
@ibatanecnica fuha tak to som nevedela teda. Ale máte pravdu, tieto okolnosti sú hrozne, to pochybujem ze by nejaký súd povolil keď zámerne nedala lieky svojej vnučke, ved to je ublíženie na zdraví.

alenah31
•
27. mar 2024
V zmysle zákona o rodine ust. § 25 ods. 5 Zákona o rodine:

Ak je to potrebné v záujme maloletého dieťaťa a ak to vyžadujú pomery v rodine, súd môže upraviť styk dieťaťa aj s blízkymi osobami(starý rodič, súrodenec dieťaťa - brat, sestra, súrodenec rodiča - strýko, teta a i.).

Vždy to musí byť v záujme dieťaťa - v tomto prípade to v jeho záujme nebude. No stará mama si návrh na súd podať môže.
AUTOR
•
27. mar 2024
Malá má 9 rokov. Manžel jej presne toto ppvedal, že stretávať sa môže ale len s našou prítomnosťou a ona sa na to začala vyhrážať súdom.

ibatanecnica
•
27. mar 2024
@lucia13022023 vieš čo, normálny nie. Ale sledujem stránku vymazaní rodičia a to, čo sa deje v našom štáte - neskutočné. ale zase predpokladám, svokra nemá také páky. Takže určite nejako neuspeje alebo len v prítomnosti rodiča

ibatanecnica
•
27. mar 2024
autorka- ktožvie, či by mala na to gule. Možno skúša vyhrážkami. A ak áno, podniknete protikroky.

gejb
•
27. mar 2024
Raz tu už tato téma bola a napíšem to co aj predtým.
Babka nech sa zmieri s návštevou vo vašej prítomnosti alebo nebude malú vidieť vôbec.
Pohrozila by som, ze ak to dá naozaj na súd, podám na ňu ja trestné oznámenie za ohrozenie zdravia alebo ublíženie na zdraví a kontakty prerusime.

matullienka
•
27. mar 2024
Byť na mieste tvojho manžela tak jej poviem, nech podá a, že on ide rovno podať TO na ňu, že ohrozuje zdravie a život vášho dieťaťa.

ibatanecnica
•
27. mar 2024
@gejb presne tak. Kristepane, čo je toto za ľudí

januska12323
•
27. mar 2024
Napíšte jej do mailu alebo sms (aby ste mali písomný dôkaz) že súhlasíte so stretávanim u vás, vonku alebo inde vo vašej prítomnosti. Vzhľadom ku ohrozeniu života dňa XY dieťa nepustite z dohľadu. A môže sa aj pokrájať.

ktoré stále sedelo na zemi, a svojou veľkou hlavou sa jej jemne šuchol o ruky. Inčučuna ho hladkala a po tvári jej stekali slzy radosti. Čierny kôň pre ňu riskoval život.

Tak dlho čakala, kým sa k nej priblíži. Tak dlho dúfala, že jej to dovolí. A teraz k nej prišiel sám. Chcel to, a dokonca ju aj chránil. Jej trpezlivosť sa vyplatila.

Ak sa dnes prejdeš po indiánskej krajine, uvidíš mladé a veľmi pekné dievča s dlhými čiernymi vlasmi, ako jazdí na krásnom čiernom koni. Nemá sedlo a ani uzdu. Len mu sedí na chrbte a drží sa jeho hrivy. Navzájom si dôverujú, rozumejú si a navzájom sa chránia.

Jedného dňa dorazili na ostrov. Neprišli tam len tak. Vedeli, že tam má byť poklad. Mali dokonca mapu. Lenže tá mapa musela byť nejaká falošná. Nedoviedla ich totižto k pokladu, ale do veľkých močiarov. Boli celí špinaví, mokrí a hlavne nahnevaní. Hľadanie pokladu chceli vzdať. Nevedeli totiž, kde poklad je. Ostrov bol veľký a nemali šancu ho celý preskúmať. Ako tak sedeli a nadávali, jeden z nich nesmelo preriekol: „ Veď máme psa. Pirátsky pes musí vedieť stopovať.“ 

Rozprávka na dobrú noc - Pirátsky pes musí vedieť stopovať
Pirátsky pes musí vedieť stopovať
Všetci otočili hlavami na Bobbyho. Dokonca aj Jednonôžka. Ten už dávno rozmýšľal nad tým, že psy stopujú. Videl to v televízií. A keby stopoval aj Bobby, ušetrili by veľa a s hľadaním pokladu. Jednoducho by prišli na ostrov a nemuseli by behať hore-dole. Bobby by ten poklad našiel. Jednonôžka nesmelo pozrel na svojho psa. Zvládol by takúto úlohu? Je malý a bláznivý, ale psy to vraj majú v sebe. Bobby vedel, na čo jeho pán myslí. Bolo by skvelé, keby našiel poklad.

Bol by hviezdou a všetci by ho obdivovali. Lenže on ešte nikdy nestopoval. Ako sa to vlastne robí? No nič vyskúša to. Bobby sa postavil a rozhliadol sa okolo seba. Všetci to brali ako znamenie, že sa chystá stopovať. Išiel vpred a všetci piráti ako húsky poslušne za ním. Bobby išiel a išiel a vôbec nevedel,  kam ide. Tváril sa však, že vie. Sem-tam ovoňal nejaký kríček a nahodil vševediaci výraz v tvári. A kráčali a kráčali.

Zrazu boli úplne hlboko v džungli a piráti začínali mať pochybnosti. To malé psíča vôbec nevie, kam idú. Zablúdili tu! Jednonôžka videl paniku v očiach svojej posádky a aj paniku v očiach svojho psíka. Zavelil na návrat. Lenže kade? Unavení piráti nadávali. Tu zafúkal vietor a Bobby zacítil vôňu gulášu z lode. On pozná cestu späť! Piráti mu veľmi nedôverovali, ale nemali inú možnosť.

Nakoniec bolo dobre, že šli za ním. Bobby ich priviedol naspäť k lodi. A to bolo dôležitejšie, ako keby našli poklad. Nič to, Bobbyho vycvičia a potom bude vedieť hľadať poklady. A zatiaľ postačí, ak bude vedieť hľadať cestu podľa vône guláša.

Rozprávka pre deti - Ľudmilka a jej dobrodružstvo v lese
Ľudmilka a jej dobrodružstvo v lese
Za chvíľu počula aj tichý hlas. „Kam som to len dala. Som to ja, ale popleta. Čo mám teraz robiť?“ Ľudmilka si kľakla za strom, aby ju nebolo vidieť, a hľadala, odkiaľ hlások prichádza. Po chvíli zbadala malú vílu. Bola celá zelená. Len hlavu jej zdobila červená čiapočka a na sebe mala červený kabátik. Poskakovala medzi papradím a čučoriedkami a donekonečna niečo hľadala. Ľudmilka si pretrela oči a myslela si, že sa jej to len zdá. Keď však malá víla prihopkala bližšie k nej, pochopila, že to nie je len sen.

Ľudmilka vykukla spoza stromu a spýtala sa: „Ahoj, víla, čo hľadáš?“ Víla sa tak zľakla, až spadla na zadok. Vôbec netušila, že tam Ľudmilka je. „Neboj sa, neublížim ti, chcem ti len pomôcť,“ ubezpečila vílu rýchlo Ľudmilka.

Víla sa najprv bála a hanbila. Potom ale spustila svojím jemným hláskom: „Som lesná víla. Dávam pozor, aby sa tu zvieratkám nič nestalo. Ale teraz som všetko pokazila. Stratila som svoj prútik. Pomáha mi, aby zimný mráz neprišiel skôr, ako by mal. Keď sa niekde objaví, a  je ešte len jeseň, prútikom ho roztopím, aby bolo všetko tak, ako má byť. Ale mráz sa už začal ukazovať a chrobáčiky ešte nie sú pripravené na zimu. Je ešte príliš skoro. Ale ja nemám prútik, aby som tomu zabránila.”

Ľudmilka neváhala, hneď začala niečo robiť. „Neboj sa, lesná víla, ja som väčšia, takže vidím viac do diaľky. Pomôžem ti nájsť prútik, aby si mohla všetko napraviť a zachrániť chrobáčikov,“ povedala Ľudmilka a začala hľadať. Poznala každé zákutie lesa, každý strom a brloh. A tak začala všetko prehľadávať. Lístok po lístku, mach po machu a strom po strome. A samozrejme, hľadala aj lesná víla.

Prešlo pár hodín a Ľudmilka prešla snáď celý les. Sadla si na zem a premýšľala, kde ešte by mohol prútik byť. Predsa sa nevyparil. Ľudmilka vedela, že od toho závisí život chrobáčikov, preto to nechcela vzdať. A potom jej ešte niečo napadlo. Vyšplhala sa na strom, kde mala straka svoje hniezdo. To je totiž vtáčik, ktorého priťahujú lesklé, trblietavé veci. Samozrejme, že prútik bol tam. Strake sa tak zapáčil, že si ho odniesla do svojho hniezda.

Ľudmilka rýchlo schmatla prútik a odniesla ho víle. Víla jej veľmi pekne poďakovala a čo najrýchlejšie začala poskakovať po lese a dávať veci do poriadku. Ľudmilka bola rada, že všetko dobre dopadlo.

Keď sa však pozrela na strom k hniezdu, uvidela straku, ako smutne sedí na konári. Bola nešťastná, že jej niekto vzal z hniezda tú lesklú vec. Ľudmilke jej bolo ľúto. Vytiahla z vrecka malé zrkadielko, ktoré sa tiež krásne lesklo. Zamávala strake a zrkadielko sa na slniečku zablišťalo. Potom natiahla ruku a ukázala strake, že si ho môže vziať. Tá priletela a šťastne si zrkadielko odniesla do svojho hniezda. Ľudmilke sa dokonca zdalo sa, že sa na ňu straka usmiala. Ľudmilka bola veľmi rada, že zase môže pomôcť lesu a zvieratkám, ktoré v ňom žijú.


Bola raz jedna mačička Micka, ktorá žila vonku na ulici. Rušné a kľukaté uličky mesta, iné zatúlané mačky či odpadkované koše bolo to jediné čo Micka poznala. A práve to bol domov sivej mačičky. No, teda nebol to domov na aký sme my, ľudia, či iné domáce zvieratká zvyknuté. Micka totiž nepoznala lásku… nemala žiaden teplý pelech, hračky či majiteľa, ktorý ju s láskou hľadí a dáva jej papať. Mala len malú papierovú krabicu, kde žila. 

Micku zobudilo krásne ráno plné slnka, ktoré jej svietilo priamo do krabice pri smetiaku, kde prečkávala dlhé noci či chladné počasie. Vykukla z krabičky, či tam náhodou nestriehne nejaký zlý pes či človek a opatrne z nej vyskočila. Ladne sa postavila, ponaťahovala si chrbátik, pooblizovala si labky a už-už sa šla vybrať do rušných ulíc veľkomesta aby si niečo ulovila na papanie, keď v tom začula hlasné: „Pomóc! Pomóc!“ 

Rozprávka na čítanie - O zatúlanej mačičke
O zatúlanej mačičke
Micka sa poobzerala po okolí, no nikde nikoho nevidela. Vyskočila obratne na smetiak pri svojej krabičke, aby sa pozrela z výšky. Poobzerala sa okolo seba, keď si v tom všimla ako malé dievčatko kričí o pomoc z okna jedného z bytov, pod ktorými Micka žije v krabici. Z okna sa valil veľký dym. To znamenalo len jediné. Oheň! Dievčatko kývalo vôkol seba malými rúčkami zo strany na stranu a do toho silno kašľalo ako volala o pomoc. Och ten dym. „Čo keď je oheň už pri nej? Musím jej nejak pomôcť!“ povedala si Micka a rozbehla sa do ulíc. 

Tam sa ľudia premávali po ulici jedna radosť a autá sa premávali kade tade. Jasné, že malé dievčatko nikto nepočuje cez ten ruch. Micka sa rozutekala a pri prechode predchodcov všimla mladú pani. Začala okolo nej cupitať a mňaučať. No pani si ju nevšímala a odbila ju jasným: „Heš!“ 

Micka smutne zamňaukala. Vzdávať sa však nechcela. Začal opäť naliehavo mňaukať a drgala do pani ňufáčikom. Tá si ju síce nevšímala, ale všimol si ju starší pán, ktorý stál vedľa nej. Čupol si k Micke, tá zamňaukala kusla ho do nohavíc snažiac sa ho potiahnuť. Potom sa Micka rozbehla do svojej uličky aby ukázala pánovi čo sa deje. 

„Počkaj mačička, kam ideš?“ kričal na Micku pán a našťastie ju nasledoval. Hneď ako vošli do uličky sa Micka vyšplhala k dievčatku, kde si sadla na parapetu okna a hlasne mňaukala. Pán si hneď všimol, že mačička sa snaží dievčatku pomôcť. Pán rýchlo vytiahol telefón, zavolal hasičov a záchranku, ktorý hneď po príchode dievčatko vyslobodili. Našťastie bola v poriadku. 

„Ďakujem Vám za záchranu!“ povedala dievčatko pánovi, ktorý zavolal pomoc. „To neďakuj mne, ale tuto kamoške mačičke, ktorá ma zavolala!“ povedal dievčatku ujo s úsmevom. Dievčatko chytilo malú Micku do náručia a vyobjímala ju od radosti. 

Mamička dievčatka sa práve vracala z obchodu, keď si všimla čo sa deje. Dievčatko jej všetko vysvetlilo a neuveríte… mačička Micka našla svoj pravý domov. Malé dievčatko bolo zdravé a v poriadku, pretože ju Micka zachránila. A dievčatko si ju za odmenu zobralo domkov. Síce  sa dievčatko s maminkou museli sťahovať, no mali novú pomocníčku so sťahovaním, záchrankyňu Micku, ktorá našla svoj domov a lásku.
Prekvapenie na šarkaniáde
Vietor im chcel pomôcť, a tak fúkal na všetky strany. Zdvíhal draky na najvyššie miesta a držal ich hore čo najdlhšie. Deti kričali od radosti. Behalii po lúke, pevne držali šnúry v rukách a mali obrovskú radosť z toho ako ich draci lietajú po oblohe. Bola to proste nádhera.

Keď bolo púšťanie šarkanov v plnom prúde, na oblohe sa niečo objavilo. Niečo rýchlo preletelo sem a tam. Mihlo sa to medzi natiahnutými šnúrami od šarkanov. Deti sa začali obzerať a rozmýšľali, čo to bolo. Nikto to však nepoznal. Až to zrazu začalo lietať medzi šarkanmi čoraz rýchlejšie a častejšie. To niečo sa zamotalo do špagátov, strhlo to takmer všetky šarkany a potom to spadlo na zem.

Všetky deti a rodičia pribehli k hromade zamotaných šarkanov na zemi. „Pozrite sa, čo tam pod nimi je! Nie, nepozerajte sa! Deti bežte od toho ďalej!“ ozývalo sa zo všetkých strán.

Hromada sa ešte viac krútila a zamotávala, až sa zrazu medzi zhlukom špagátov, špajdlí a rôznych farieb objavil malý ňufáčik. Všetci odskočili. Po chvíli sa z hromady vyhrabal malý drak. Pozeral na všetkých a nechápal, čomu sa čudujú.

„Ahoj, ja som dráčik Darebáčik. Počul som, ako všetci hovoria o šarkaniáde, a chcel som sa tiež pozerať a zalietať si. Ale nevedel som, že to budú vyrobené šarkany draci. Myslel som si, že tu budú lietať skutoční draci. Nehnevajte sa, že som vám to pokazil a zamotal som sa do vašich drakov,“ smutne vysvetľoval malý dráčik.

Rodičia sa najprv na malého tvora neveriacky pozreli. Deti však boli nadšené. Vôbec im nevadilo, že ich draci sú polámaní, tešili sa, že stretli skutočného draka. Dráčik Darebáčik im vysvetlil, že žije v jaskyni v neďalekom lese. Nikto ho nikdy nemohol vidieť, pretože lietal len v noci a potichu. Často bol sám a bál sa ľudí. Keď sa však dopočul o šarkaniáde drakov, myslel si, že sa stretne s inými drakmi. Spočiatku bol smutný, že je naozaj jediným drakom. Potom však zistil, že v skutočnosti nie je sám. Skamarátil sa s deťmi a ich rodičmi.

Nielenže potom deťom pomáhal a sprevádzal ich do školy, ale aj keď sa konala ďalšia dračia šarkaniáda, veselo lietal medzi papierovými drakmi a snažil sa deťom udržať drakov na oblohe čo najdlhšie.
Čarovný les
Všetci sa v lese mali radi. Len víla Lesnička bola dnes smutná. Ako letela ponad les, zbadala niečo čo ju veľmi rozplakalo. Ani štvorlístok jej náladu nezdvihol.  „Čo sa stalo?“ spýtal sa škriatok Lesničky. Tá smutne sedela na stebielku trávy a nariekala. „Niekto vyrúbal naše stromy…a chce tu postaviť..obchody!“ zvolala smutne a ukázala na kraj lesa.

„Ó nie! Kto?? Kto to tu vyrúbal?!“ zvolal nahnevane škriatok.

Nebol to však hocikto.. boli to veľké hlučné stroje…ktoré riadili stvorenia, ktorých sa magické bytosti a zvieratá najviac báli..boli to ľudia. Ľudia im prišli zničiť krásny les! Ich prírodu!

 To predsa nemôžu nechať len tak. „Neboj sa, my naše stromy a náš les zachránime!“ škriatok a spolu s vílou Lesničkou sa vybrali za múdrou sovou Vedmou. Tej všetko porozprávali.

„Ako len náš len zachránime?“ spýtala sa Vedma zvieratiek a čarovných bytostí na ich poobednom stretnutí. „Začarujeme ľudí!“ zvolala víla Kvetinka. Sovička len pokrútila hlavou. „Tak im dáme vypiť elixír!“ povedala čarodejnica Olga.

„Nie, nie, na ľudí čary použiť nesmieme..to je zakázané. Les by sa nahneval.“ Povedala Vedma. „Tak ich zjeme!“ zvolal mocný vlk a od radosti zavil. Sova sa prísne na vlka pozrela a jeho nápad hneď zahriakla. „Nikoho jesť nebudeme!“ povedala. Zvieratká a všetci ostatní sa zamyseli…čo by len vymysleli..ako svoj les zachránia…keď v tom dostala Lesnička nápad.

„Vyplašíme ich! Vyplašíme ich tak, že do nášho lesa už nikdy viac neprídu!“ zvolala hrdinsky.

A tak aj bolo. Všetky zvieratá a aj bytosti spojili sily a vybrali sa na koniec lesa. Tam  sa práve drevorubači chystali zoťať ďalší strom, zatiaľ čo iný človek mal v ruke plány. Chceli len premeniť na obchodné centrum. Keď v tom sa ich papiere s plánmi zdvihli do vzduchu. Ľudia na seba prekvapene pozreli. Ich stroje sa začali dookola točiť. Nevedeli to zastaviť. Veselí škriatkovia fúkali na stroje svoje štvorlístky a tým stroje točili.  Zatiaľ čo vlci silno vyli a medvede sa pomaly blížili,  čarodejnice vytvorili strašidelné zvuky. Ľudia sa tak zľakli, že vyskočili zo svojich strojov, pozbierali svoje veci a utekali preč. Dokonca vyskočili aj z tých čo sa točili ako na kolotoči! A neuveríte … do čarovného lesa sa už nikdy nevrátili a všetci boli v bezpečí. A práve preto, že všetci spolu držali a pomohli si, zachránili svoj les!
„Ahoj! Nemusíš sa ma báť…. pomôžem ti?“ spýtal sa milo Brumík.

 Vtáčik, ktorý sa volal Bobík sa ostýchal, no keby ho chcel medveď zjesť už to dávno spraví.. preto nakoniec odfrkol: „Zachytil som sa do týchto konárikov. Neviem sa z kríku dostať von! Moje krídla sú zamotané!“ zabedákal malý vtáčik Bobík.

Brumík neváhal a hneď vtáčikovi pomohol. Ten sa postavil na malé nôžky no vyletieť nedokázal. Jedným krídlom síce vrtel ako o život, no druhé ho akosi neposlúchalo. Nevedel vzlietnuť. Pošuchoril si perie, no i tak nič.

„Akosi som si poranil svoje krídlo v tom kríku,“ povedal smutne vtáčik.

Medvedík navrhol Bobíkovi, že sa o neho postará. Ten nakoniec súhlasil. Brumík opatrne položil vtáčika do svojho košíka a vybrali sa do brlohu. Tam medvedík opatrne očistil krídelko vtáčikovi medíkom.
Hlboko v lese vo veľkej nore žila rodina jazvecov. Otec jazvec, mama jazvečica a ich dve neposedné krásne mláďatá. Boli to dvaja jazvecí chlapci. Volali sa Monty a Piškót. Obaja boli veľmi dobrí. Pomáhali svojim rodičom so všetkým. Stavali s nimi zložité nory a zháňali potravu po celom lese. Ich rodičia z nich mali veľkú radosť.

Ale Monty a Piškót boli tiež plní energie a stále na niečo mysleli. Radi sa hrali na naháňačku a na schovávačku a boli dobrými priateľmi s ostatnými zvieratkami. Boli to odvážni chlapci a ničoho sa nebáli. Jedného dňa sa však predsa len stalo niečo, čo ich vystrašilo.  

Bolo krásne slnečné popoludnie a Monty a Piškót už mali všetky svoje povinnosti splnené. Rozhodli sa prebehnúť po lese, nájsť veveričky a zahrať si s nimi nejakú hru. Prechádzali sa po lese krížom krážom. Volali na svoje kamarátky, ale nemohli ich nájsť. Tak išli ďalej hlbšie do lesa.
Bol raz jeden hnedý psík Hafik, ktorý žil vo veľmi dobrej rodinke. Bol to malý psík no odvahy mal neúrekom. Každý deň dostával plnú misku granuliek, vodičku a vždy sa tešil na prechádzky v blízkom parku. Tam sa veselo hral i s inými psíkmi či naháňal neposedné veveričky. Vždy keď jeho najlepší kamoš, chlapec Martinko prišiel autobusom zo školy domov, Hafík ho už verne čakal pri bráne, pripravený na ďalšie dobrodružstvá. Najviac sa Martinko a jeho psík ľúbili hrať na superhrdinov. Vždy niekoho „akože“ zachránili pred zlými zlodejmi, či lietali ako statočný Superman.   

Jedného dňa, keď sa Martinko vracal zo školy, jeho verný psík Hafik pri bráne nestál. Čo to čo to, veď kde je jeho verný priateľ? … Martinko nevedel čo sa deje. Zahodil školskú tašku na zem a utekal rýchlo dnu do domu. No ani maminka ani ocinko Hafika nikde nevedeli nájsť. Doma nebol. Kričali ako len vedeli, no ich verný psík neprichádzal. Dokonca ani pod paplónom nebol ..a tam sa Hafík ľúbi schovávať najviac. 

„Skúsim ešte garáž!“ zvolal chlapček, keď ho doma nevedeli nikde nájsť. V garáži natrafil na malého pavúčika, ktorý sa schovával v rohu miestnosti. 

Rozprávka na čítanie - Hafík sa stratil
Hafík sa stratilMilé detičky určite si spomínate na psíka Hafíka, ktorý žil u Martinka doma. Boli to verní priatelia, ktorý spolu vymýšľali rôzne šibalstvá. Najviac sa však ľúbili hrať na superhrdinov. Hafík vždy Martinka čakal pri bráne, keď prišiel zo školy aby sa šli hrať či už na Supermana alebo Batmana. No jedného dňa tam Hafík nestál a Martinko sa veľmi bál. Všade ho hľadal či už doma, v garáži ale i na záhrade. No psík nebol nikde. Počas hľadania natrafil na pána pavúčika, pani veveričku i na pani včielku.

Všetky zvieratká mu poradili, kde psíka videli naposledy. Zhodli sa na tom, že psík utekal von zo záhrady. Martinko preto poďakoval zvieratkám za ich rady a vybral sa hľadať psíka von na ulicu, kde už hľadali aj jeho rodičia.
Vojtíkov starý otec sedel vo svojom kresle a díval sa z okna, ako vonku prší. Okolo sa obšmietal

zamračený Vojtík a nevedel, čo má robiť. Tešil sa už na veľkonočnú nádielku, ale do tej zostávalo ešte toľko dní.

„Nemrač sa Vojtík, je Škaredá streda. Zostane ti to,“ smeje sa starý otec.

„Prečo škaredá? Pretože prší?“ pýta sa Vojtík.

„Nie, nie. Škaredá streda patrí do veľkonočného Svätého týždňa. Alebo si myslíš, že je Veľká noc len v pondelok?“

Vojtík sa prestane mračiť a so zaujatím sa pýta: „Tak prečo je tá streda škaredá, dedko?“

„Je škaredá, zamračená. Než ti poviem prečo, musím ti vysvetliť, prečo sa vlastne oslavuje Veľká noc. Vieš, kto bol Ježiš?“

Rozprávka pre deti Vojtíkova Veľká noc Škaredá streda
Vojtíkova Veľká noc: Škaredá streda, Anička I.
Bol piatok pred VeľNočné kysnutie + dievky, ktoré v dňoch voľna zásadne vstávajú s prvými lúčmi slnka = výborné raňajky 🥰



oxiba
Juj, tak to vyzera mnam. Hned by som si dala🙂
28. mar 2024

nesvadbovo
AUTORAMBASÁDORKA
@oxiba skoricou sa skratka neda nikdy nic pokazit 😉
28. mar 2024

jany149
Vyzerajú mňamkovo 👍
31. mar 2024

audrye
Mozem poprosit recept na cesto na nocne kysnutie? (: ď.
31. mar 2024

zuzana1zuzana
📌
31. mar 2024

ivula251
@nesvadbovo ahoj , prosím pridaj recept 😉
31. mar 2024

nesvadbovo
AUTORAMBASÁDORKA
@jany149 vdaka, naozaj boli 😉
31. mar 2024
Existuje práca kde nieje žiaden alebo kde je minimum stresu?

Strana

z3

ptvw
•
27. mar 2024
Podla mňa nie
Ešte doplním, aj keď ma niekto záľubu ako pracu, vždy sa isto stane, že príde k momentu, keď ta niečo niekto tlaci a už to ide

doriiis
•
27. mar 2024
Vo výrobe....keď nie si úplný gramblos a keď sa naučíš,potom je to len rutina

larinka552
•
27. mar 2024
V nejakom malom obchode, stánku, v turistickom centre, cestovnej kancelárii. Tam kde sa točí málo ľudí.

obkladac
•
27. mar 2024
Stres väčšinou vzniká pri nedostatku času a informácií ak tieto dve veci vynecháš máš ideálnu prácu.

drvanko
•
27. mar 2024
Ahoj no stresy sú aj budu mimo práce stresové situácie máme každý druhy človek aj domáca napätá atmosféra nervy a tak td. ja radšej idem do kaviarne sa vyhýbam takých to stresových situácií a čo preživáš stresové situácie?...ľudia to držia v sebe a potom nemajú na koho vybiť zlosť tak vybiju si zlosť svoju na obyčajného človeka

simiik12
•
27. mar 2024
Ja pracujem v Domove sociálnych služieb ako sociálny pracovník a stres není 🙂

denkabemommy
•
27. mar 2024
Pracujem v marketingu a mám to na 99% bez stresu 🙃. Chodím si do práce oddýchnuť.

nicicka
•
27. mar 2024
Skor si myslím ze je to o povahe človeka ako o charaktere práce, pripadne kombináciou oboch. Jeden môže určitý druh práce zvládať ľavou zadnou no pre druhého to môže byť nočná mora. Radšej by som makala na sebe, vyskúšala techniky na zníženie stresu, vzdelavala sa v danej oblasti a stres sa potom prirodzene zníži.

buberova
•
27. mar 2024
@doriiis what??? Želám veľa šťastia s normami alebo úkolovou mzdou

brrunetkaa
•
27. mar 2024
Štátna správa. Administratíva, ktorú sa nauči aj cvičená opica. 8 hod padne a ideš domov bez nejakej ďalšej záťaže. Akurát že vyžadujú II VŠ stupeň (aspoň na pozícii kde som ja) aj keď netuším načo…asi len aby dobre menovka vyzerala 😀

janulik2008
•
27. mar 2024
@doriiis pracovala si vo výrobe? Pretože @buberova ma pravdu

oli18
•
27. mar 2024
@simiik12 co znamena socialny pracovnik?

doriiis
•
27. mar 2024
@buberova ...myslím výroba ako automobilka alebo dodávateľské firmy.Prides,odrobis,ideš domov.

janulik2008
•
27. mar 2024
@doriiis není to také jednoduché že prídeš, odrobiš a ideš domov. Čo sa týka času tak áno ideš na určitú hodinu a o osem hodín končíš lenže práve to "prídeš odrobiš" je trz u nás že sú reklamácie a šli "hasiť" aby nám nevzali projekt.

doriiis
•
27. mar 2024
@janulik2008 ....no keď si na vyššej pozícii tak možno áno.Ale obyčajný operátor v automobilke len doslova odrobi jak cvičena opička a ide domov.Nikto ho neposiela nikam nič hasit

dollyzv
•
27. mar 2024
@doriiis ten klasický operátor má normu ktorú musí za daný deň spraviť..a keď pochybi,reklamácia sa týka samozrejme aj jeho a môžu mu stiahnuť odmeny napr. Aspoň u nás to tak bolo keď som robila vo vyrobe..ešte dodám,že napr. U nás stroje nefungovali vždy ako mali a normu si samozrejme tým pádom nestihla,alebo si ju musela dobehnúť,keďže ten čas opravy ťa zdržal..u nás také ako prestojne nebolo

roxanna
•
27. mar 2024
Myslím si, že napríklad v knižnici, zoologickej alebo botanickej záhrade, múzeu, galérii.

buberova
•
27. mar 2024
@doriiis a teda robila si vo výrobe? Alebo niekto z tvojich blízkych? Lebo mám vo svojom okolí takých ľudí a tí proste idú ako píly... Pekne na úkol, takže žiadne zašívanie sa ani flákanie, normy sú stanovené, ak ich nesplníš dostaneš minimálku... A určite tie normy nie sú nastavené na pohodové tempo.. takže robíš s pocitom, že nestíhaš, bolí ťa chrbát/krk/zápästie, nevadí Málaš ďalej. Ak sa nájde 1 chybný kus, 1 reklamácia, všetko čo si vyrobila z tej várky si ideš pekne krásne skontrolovať. Kus po kuse. Nestíhaš potom normu, do úkolu sa ti to tiež neráta, takže niekedy aj na úkor obedu... Áno, nemusíš sledovať zákony a mať zodpovednosť za celé oddelenie alebo za to, či ten človek bude ešte niekedy chodiť, takže v tomto zmysle nemáš stres.. ale si pod tlakom aj vo výrobe, a kto tvrdí opak buď natrafil na veľmi dobrú firmu, alebo nemá reálnu skúsenosť "len si to tak myslí."

doriiis
•
27. mar 2024
@buberova ...áno,pracujem v automobilke už 8 rokov.Ano,to,čo vy s @dollyzv popisujete môže tak fungovať v niektorých firmách.Konkretne hovorím za tú automobilku v ktorej pracujem,že žiadnemu operátorovi sa nesahaju na prémie iba preto,že niečo zle vloží,alebo mu spadne alebo nestíha. Max tak zastaví výrobu.Nieco také ako ukolova mzda u nás neexistuje a normy sa nestihaju úplne bežne a väčšinou to nie je chyba človeka.
Ale áno,chapem vás.Pracovala som aj pre menšiu firmu v rámci automobilového priemyslu a tam mali takéto zvyky.Ale tiež nemali ukolovu mzdu a max im tak siahli na osobné prémie.Aj to zriedkavo.
Nie je výroba ako výroba a nie je zamestnávateľ ako zamestnávateľ.
Ale ano

lindous
•
27. mar 2024
Tiež si myslím, že veľmi záleží aj od povahy človeka... ja som mala brigádu, ktorá podľa mňa bola úplne bez stresu, jednoduchá administratíva, pomoc s triedením dokomentov, archiváciou atď. proste úplná pohodička, nikto ma nenáhlil, prišla som odrobila som a pustila z hlavy, keď sa niečo nestihlo spravila som to na ďalší deň, nejaký veľký priestor na chyby tam nebol a napriek tomu som mala kolegyňu, s rovnakou náplňou práce, ktorá to prežívala akoby sme boli kardiochirurgičky a sama seba stresovala 😅

tibor79
•
Autor odpoveď zmazal
•

tibor79
•
27. mar 2024
@simiik12 soc.pracovnik to nemá ľahké, neustály kontakt s klientmi, príbuznými, manažovať spolupracovnikov. Práca s toľkými ľuďmi denno-denne je náročná.

majlo1234
•
27. mar 2024
Stres je aj otazka povahy, ja moju pracu povazujem za nestresovu, projektovy manazment povacsine, su momenty, ked sa nieco poserie, ale to asi vsade. Inak ked clovek vie, co robi, nema tolko stresu.

ivangeline
•
27. mar 2024
Bez stresu by si sa nikam neposunula... Aj kebyze robis sama na seba,stres je pozitivny a posuva dalej. Aj ked robis na svojom,tak mozes mat stres,lebo vzdy su nejake pravidla,casy,rezim,nutne cinnosti..

ysseba
•
27. mar 2024
@brrunetkaa Ja tiež pracujem v štátnej správe a moju prácu by sa cvičená opica nenaučila. Stresových situácií je viac ako dosť, stále termíny (na tie sa ale zvyknúť dá) a audity, na tie sa zvyknúť nedá a za všetkým čo spravím je otázka "čo ak si to vyberie audit a ako to bude brať".

janinah
•
27. mar 2024
Existuje a ja ju budem robit v budúcom živote,budem kvetinarka.

janulik2008
•
27. mar 2024
@doriiis no, ja už v piatok bude 8 rokov čo robím v jednej firme, operátor a veru že ako tu niekto spomínal sťahujú prémie keď napr niečo nájdu (ale to už je naozaj posledná možnosť). Skoro 8 rokov som kontrolovala kožené diely do áut a boli reklamácie do "aleluja" len my sme sa dostali už trz do takej "šlamastiky" že manažér a riaditeľ šli na služobnú cestu aby zachránili čo sa dá, aby projekt nevzali, lebo sme v prdeli všetci. Posledné mesiace už som na inej pozícii, striham na strojoch s tam tiež to nie je niekedy sranda, stačí že do stroja zle zadáme niečo a už stroj vystrihá úplne iné kusy (dnes sa stalo kolegovi), lebo musel prepínať stroj niekoľkokrát lebo ma Audi sa používa taká koža a na Mercedes iná.

svetlonos
•
27. mar 2024
@janulik2008
tak to je masaker tie kozene diely. My ame boli pozriet z kancerarii do fabriky, kde siju kozene poťahy. Normalne sa mi nozik vo vrecku otvaral, ako tam s ludmi jednali. Nad kazdym pracovnym stolom obry displej, aka je norma a kolko kto mešká. Na tabuliach vypisany najleosi a najhorsi zamestnanec mesiaca - pre motivaciu ... 🤯 plus ine fajnovoty. Tie krajcirky tam maju bud zlate ruky alebo oci pre plac.

katka700
•
27. mar 2024
Práca bez stresu neexistuje, všade je dačo

bluka2
•
27. mar 2024
@svetlonos Najhorší zamestnanec mesiaca? 🤯 to sa vôbec môže…? Drsné. A čo sa tomu človeku potom stane, keď vyhrá tu cenu povedzme 3x?

nesvadbovo
AUTORAMBASÁDORKA
@audrye uplne hocijake, ake si zvyknuta robit. V tomto pripade kyslo klasicky na linke cca hodinku, potom som ich zvinula a dala do chladnicky kysnut do rana (cca 8 hodin), rano rovno z chladnicky vkladam do rozohriatej rury..
31. mar 2024

nesvadbovo
AUTORAMBASÁDORKA
@ivula251 inspiraciu som brala odtialto, ale vynechala som tentokrat pomarance a lekvar, boli cisto skoricove. V podstate mozes pouzit aj klasicky recept na osie hniezda, len ich inak zatocis..
https://stvoryzkuchyne.com/skoricove-uzliky-s-d...
31. mar 2024

audrye
@nesvadbovo ďakujemkou nocou a Vojtík už sa tešil, ako pôjde popoludní za dedkom vypočuť si ďalšiu časť príbehu Veľkej noci. Ako sa ale čudoval, keď po obede prišiel starý otec za ním.

„Ahoj Vojto, rýchlo obliekať. Dnes je Veľký piatok, tak nech nám tie poklady niekto nevyberie,“ volal starý otec od dverí s úsmevom.

„Aké poklady, dedko?“ pýtal sa Vojto a hľadal topánky.

„Hovorí sa, že na Veľký piatok sa otvárajú pukliny v skalách alebo diery v zemi. V nich čaká na poctivých ľudí poklad.“

Rozprávka pre deti Vojtíkova Veľká noc: Veľký piatok
Vojtíkova Veľká noc: Veľký piatok
„Jej, vážne? Ja ho chcem nájsť,“ volal Vojtík a už sa súkal do bundy.

Za pár minút si to Vojtík už šinul po ceste k lesu popri dedkovi.

„A ako ten poklad spoznáme, dedko?“

„Hovorí sa, že bude žiariť. Ale je to len povera, Vojtík,“ dodal starý otec.

Slniečko pekne svietilo a hrialo Vojtíka do tváre.

„Pozri, tam niečo svieti!“ volal Vojtík a vrhol sa do diery od vyvráteného stromu.

Ale to len presvitalo slniečko. Vojtík sa chvíľu prehraboval v hline, či predsa len nejaký poklad nenájde, ale nič tam nebolo. Išli teda ďalej. 

„Ja myslím, že tamto niečo je,“ povedal starý otec a ukázal na hromadu balvanov pri lesnej ceste.

Vojtík sa tam hneď nadšene rozbehol. Však tu to poznal. Sem sa chodil s dedom hrať dosť často. Aké pre neho ale bolo prekvapenie, keď medzi kameňmi objavil malú drevenú truhličku a v nej niekoľko čokoládových a niekoľko pravých mincí.

„Dedo, dedo! Našiel som poklad!“

„Vážne?“ čudoval sa starý otec.

„Pozri,“ radoval sa Vojtík a ukazoval dedkovi poklad. „Ty si o ňom vedel, že? Dal si ho tam ty?“
Ahojte baby.
Mám na vás otazku. Nájde sa tu nejaká žienka čo reálne schudla do mesiaca Max do dvoch? Mne sa absolútne nedarí od 1.1. som začala cvičiť prestala som piť sladené nápoje na ktorých som dlhé roky "závislá" prestala som jesť sladkosti a pečivo....
1.3. som sa zrútila začal mi chýbať strašne cukor nervy stresy v robote atď.... V lete idem na svadbu a celkovo si ani neviem predstaviť ísť si dať plavky a ísť na kúpalisko.... Mám 70kg a 167cm je to moc necítim sa dobre samozrejme že nechcem ležať doma a čakať na zázrak cvičenie mi nevadí len trošku nejakú pomôcku nejakú radu prípadne lebo fakt som stratená
Keto diétu som skúšala ale to je hnus extrémne drahé a hlavne hnusne.Prosím ak je tu niekto kto má podobnú skúsenosť a poradí budem veľmi vďačná potrebujem schudnúť do 10 kg ďakujem 🩷 a za každú radu vopred ďakujem


thereisabear
•
Dnes o 17:03
Kalorické tabuľky.
A keto sa dá jesť aj bez toho hnusu, s normálnym jedlom.

viktoriaviktoria11
AUTOR
•
Dnes o 17:17
@thereisabear no tie kalorické tabuľky popravde z nich som úplný debil.... 🤷🏾 Nerozumiem tomu nejako keďže som to nikdy nepoužívala

januska12323
•
Dnes o 17:17
70 kg a 167cm je dosť ale nie až tak extrémne veľa. To by mohlo stačiť zdravo jesť a nejak cvičiť.

viktoriaviktoria11
AUTOR
•
Dnes o 17:20
@januska12323 veru áno je to dosť z každej strany mi to je pripominane že som pribrala už to nevládzem počúvať 😐..... Cvičenie mi nerobí problém len keby niečo aspoň čo eliminuje chuť do jedla alebo niečo také lebo niekedy je to fakt taký stav že sa nedá odolať vobec ničomu už fakt neviem ako mám začať svoju váhu mám 60-65 kg len mám to kolísavé kôli štítnej to

thereisabear
•
Dnes o 17:24
@viktoriaviktoria11 nemáš facebook? Tam je k tabuľkám dobrá skupina. Nie je to vôbec zložité a funguje to.

ninka99865
•
Dnes o 17:28
@viktoriaviktoria11 a štítnu máš nastavenú liečbu?
Ja ti odporúčam nájsť si osobnú trénerku/ trénera ktorý ti presne nastaví aj jedálniček. A ty si len budeš veci vážiť a zapisovať do tabuliek

lindous
•
Dnes o 17:31
Pri dobre nastavenej vývaženej strave si môžeš dať aj ten koláčik alebo pohár coly k obedu a nemusíš surovo vyškrtnúť všetko čo máš rada, ale považuješ to za “zlé”… potom ťa ani chute tak nebudú presnasledovať, nebudeš mať výčitky a pocit, že keď už si to raz “pokazila”, tak sa môžeš “obžrať” lebo už je to jedno a celý tento kolobeh dookola… začať s kalorickými tabuľkami je fajn, aj tu je skupina “Cesta k mojej premene”, kde je ti fajn vysvetlené a kočky sú ochotné pomôcť 😊 dostaneš sa do toho určite, keď si niečo prečítaš a potom aspoň pár týždňov nahadzovať, aby si zistila kde sa asi pohybuješ s príjmom, čo má aké živiny a koľko kalórií… ak sa v tom cítiš úplne stratená, nevieš ako začať s cvičením atď, tak by som zvážila dobrého a naozaj vzdelaného trénera/trenerku 😊

viktoriaviktoria11
AUTOR
•
Dnes o 17:33
@ninka99865 Úprimne nemám ešte pred koronou mi ju zistili a mala som ju úplne slabucku takže mi ani nechcel dať lieky len mi chcel spraviť protilátky dostala som silný covid a odvtedy som tadiaľ ani nešla....nebudem sa vyhovárať vykašľala som na to....keby som skôr vedela ako má to hormonálne bude deptat a pod už to dávno riešim objednala som sa teraz ale objednali má na najbližší termín 8.8 😩

nika11103
•
Dnes o 17:33
@viktoriaviktoria11 ja mám 174 cm a vytiahla som to na solídnych 92 kg To je len nadváha . Ja skôr nejem ako jem ale myslím že chyba pohyb a jak tu budú písať baby alebo písali proste neoklameme to musíme mat vyšší vydaj ako príjem . A ja do toho si rada vecer dám pohárik a niekedy aj dva proseca . A to je asi tá najväčšia škodná . Snáď má tu neoznacia za alkoholicku , dám si fakt pohárik ale niekedy dva vecer keď všetci zaspia a idem spat aj ja . Nie je to každý vecer ale dvakrát do týždňa a niekedy aj tri určite

viktoriaviktoria11
AUTORAhojte, píšem anonymne lebo..
Niekto mne blízky zvolil smrť, tragickú smrť.
2 dni som brala Neurol, už ho nechcem.
Ale potrebujem sa skľudniť, neviem existovať, neviem spať, milión otázok v hlave- odpovede žiadne.
Psychológa zatiaľ nepotrebujem- ak by bolo treba, vyhľadám ho.
Ďakujem za rady.


drvanko
•
Dnes o 16:50
Ahoj skús odbehnúť ku neurologičke ona mne predpisuje Diazepam 5mg on je dobrý na uvoľnenie,stresu,depresií,úzkosti,nepokoji,Nechoď ku psychiatričke lebo mne predpísala liek a bolo mi zle stoho

deti95060810
•
Dnes o 16:53
diazepam a neurol je jedno a to isté... oblbovák na ktorý si pomerne rýchlo zvykneš...😉
@drvanko
AUTOR
•
Dnes o 16:54
@drvanko ďakujem, ale diazepam nechcem, to je tiež niečo, na čo sa dá rýchlo zvyknúť.
Ale ďakujem i tak za radu.

deti95060810
•
Dnes o 16:57
https://www.adc.sk/databazy/produkty/podobne/ne...
tu sa dočítaš že Neurol a diazepam sú veľmi podobné lieky... podobnosť 78 perc.

drvanko
•
Dnes o 16:59
@deti95060810 ale ja som bral neurol a mi bolo zle blúznil som a motalo ma

deti95060810
•
Dnes o 16:59
skús väčšie dávky magnézia....
len či pomôže neviem...
ono zas za týždeň - dva si nikto nenavykne na Neurol. to by si musela brať omnoho dlhšie.
Môže sa stať že po týždni - dvoch ten stres z teba opadne a neurol už nebudeš tak potrebovať... Takže ak máš zlé stavy - asi bude potrebné neurol brať za istý čas.

deti95060810
•
Dnes o 17:00
@drvanko akú dávku si bral? to čo píšeš si mal asi veľa...

drvanko
•
Dnes o 17:01
@deti95060810 nošak oblbovak zato neberiem ...diazepam on mi pomáha nesom taky celý napätý

deti95060810
•
Dnes o 17:01
@drvanko aký a koľko si bral Neurolu za deň... mal si dávku zmenšiť... niekedy stačí 1/2 tabl.
AUTOR
•
Dnes o 17:03
@deti95060810 to viem, že za týždeň- dva sa závislosť nevytvorí ( teda dúfam), ale ani mi nesadol.
Teda neviem či to tak má byť, ale bola som ako v inej dimenzii, a k tomu ma ešte bolel žalúdok.
Magnézium zvýšim, to je to najmenej.
Ďakujem.

miss_foxy
•
Dnes o 17:03
Mne na stres pomohol Anxiolan. Ale možno tebe bude slabý :( skús sa poradiť v lekárni

drvanko
•
Dnes o 17:03
@deti95060810 ja mam tak že si ho lomím na polku a pred spaním si ho dam a potom v noci ešte ten Anxiolan ten je vyživový doplnok veľa ľudí na modrom koníku sa mi pýtali dievčatá či to môžu uživať a či to je na predpis vysvetloval som im že to je ľahko dostupný Anxolan bez predpísu

deti95060810
•
Dnes o 17:05
@drvanko ja som len chcela že aký neurol si mal ...lebo je 0,25 ,
0,5 a 1,0 ....

drvanko
•
Dnes o 17:06
@miss_foxy Ahoj presne i ja ho užívam dávam si ho v noci okolo jednej najprv Diazepam polku a potom Anxiolan....Medovka niekedy mi pomáhala tiež ale už mi je nejako zle po Medovke

drvanko
•
Dnes o 17:06
@deti95060810 neurologička ti predpisuje?...
AUTOR
•
Dnes o 17:08
@drvanko chlape, však čítaj kto ti čo píše.
Splietaš tu piate cez deviate.

drvanko
•
Dnes o 17:10
ľudia ja už myslím asi od 2016 roku užívam ten Diazepam ja si myslím že už si zvyklo moje telo na ten liek asi tak no

deti95060810
•
Dnes o 17:11
@drvanko kdeže... obvodná....

drvanko
•
Dnes o 17:12
prepáč,ale keď nestíham tuto čítať aj písať ľudom

drvanko
•
pribinak
•
Dnes o 16:01
@kajka229 otec? Babka? Krstna?

kajka229
AUTOR
•
Dnes o 16:08
@pribinak otec o mňa nemá zaujíma starká pred nedávnom mi zomrela a krsntnu mám ale ona má už veľa rokov a má toho veľa starosti

30katrusa40
•
Dnes o 16:11
@kajka229 internát ? Aspoň v týždni by si mala ako tak pokoj. Prípadne skus ísť na úrad práce tam by ti mali vedieť poradiť.

pribinak
•
Dnes o 16:13
@kajka229 veľa rokov je koľko?

solange7
•
Dnes o 16:13
@kajka229 nemas aspon tetu alebo stryka? Ak nie, tak na tvojom mieste by som sa zdoverila mamine nejakej tvojej dobrej kamaratky. Ked ozaj nemas komu, mozno by ti trochu odlahlo sa s niekym porozpravat.
Modrý koník
•
Dnes o 16:16
Ahoj,

je nám ľúto, že vo svojom živote prežívaš zlú situáciu. Je potrebné o problémoch hovoriť, nemlčať a požiadať o pomoc.

Pomocnú ruku Ti bezplatne podajú na týchto linkoch, prípadne telefónnych číslach:
https://www.modrykonik.sk/faq/z-mojej-zivotnej-situacie-uz-nevidim-ziadne-vychodisko/

Veľa sily a pekný deň
Modrý koník - Sprievodca tehotenstvom a materstvom

kajka229
AUTOR
•
Dnes o 16:17
@30katrusa40 na internáte nie som

kajka229
AUTOR
•
Dnes o 16:17
@pribinak 70

pribinak
•
Dnes o 16:17
Máš v škole psychologickú? Obrat sa na ňu a ona môže kontaktovať kuratelu

pribinak
•
Dnes o 16:18
@kajka229 však to je v pohode. Ty nie si malé decko

kajka229
AUTOR
•
Dnes o 16:18
@solange7 tetu mám aj sesternicu ale jej to nechcem povedať

kajka229
AUTOR
•
Dnes o 16:21
@pribinak nechcem ísť do deckeho domova

kajka229
AUTOR
•
Dnes o 16:21
@pribinak čo je pohode?

solange7
•
Dnes o 16:23
@kajka229 mas 16 rokov. Si dost stara na to, aby si vedela, ze odist sa len tak lahko neda. Nemas prijem, predpokladam ze este chodis na strednu a je skoda to zahodit aby si sa niekde pretlkala.
Najlepsie bude aj sa zdoveris tete. Ked mas strach z toho ze by si nastrbila vztahy medzi tvojou mamou a tetou, to by som neriesila. Mozno prave tymto sa mama trochu pozbiera a spamata.

pribinak
•
Dnes o 16:23
@kajka229 však nemusíš ísť do detského domova, bude sa hľadať medzi príbuznými kto by sa tie 2 roky o teba postaral

pribinak
•
Dnes o 16:24
@kajka229 vek 70 rokov. Tam by lekár zhodnotil ci je teta schopná sa posrata

cacianka
•
Dnes o 16:26
Je mi ľúto, čo zažívas. A komu vlastne to povedať chces ? Keď o rodine píšeš že nie.
Školskej psychologičke ? Triednej Učiteľke? Lekárke ku ktorej chodiš ?
Zájdeš na upsvar- odbor pre deti a mládež ? Oni by ti mali pomôcť najviac, riešiť pohovor s mamou ak ťa zanedbáva, riešili by pohovor s jej obvodnou lekárkou aby mamu vyšetrila, mamu by usmerňovali, pozorovali by či sa stav zlepšuje, mali pracovať a smerovať mamu k lepšej starostlivosti, lebo hrozí že jej ťa odoberú.
Si zmierená, že by si šla aj do náhradnej rodiny, prípadne do domova pre deti ?

kajka229
AUTOR
•
Dnes o 16:27
@solange7 odísť je není ľahké nejakí podnájom ale peniaze nemám

kajka229
AUTOR
•
Dnes o 16:29
@cacianka neviem asi nikomu nechcem to povedať len niekde odísť ale peniaze nemám na podnájom

kajka229
AUTOR
•
Dnes o 16:30
@pribinak určite krsnej to nebudem vravieť

3009
•
Dnes o 16:32
@kajka229 a prečo to nechceš povedať krstnej? Určite by ti pomohla.

kajka229
AUTOR
•
Dnes o 16:36
@3009 no ona má svoje problémy

cacianka
•
Dnes o 16:36
Nechceš to nikomu povedať, nemáš peniaze, nie si plnoletá - nemáš iné možnosti, ako to čo som ti napísala.
Mama potrebuje nejakého človeka, ktorý by mal k nej autoritu, dohovorila jej, pomohol by ti, ale ty všetko a všetkých neguješ. Tak ako ti niekto môže pomôcť, keď ty pomoc nechceš / odmietaš ?

kajka229
AUTOR
•
Dnes o 16:42
@cacianka Ja chcem odnej odísť

solange7
•
Dnes o 16:43
@kajka229 niesi plnoleta. Nemozes ist do ziadneho podnajmu. Okrem ineho keby si chcela ist do podnajmu, tak si mozes pripravit tak 400e mesacne a to si este nejedla. Asi si teraz v zlom rozpolozeni, za to ze ta mama zbila ale poviem ti narovinu, premyslat nad takymto radikalnym riesenim je zbytocne. Bud sa zdoveris rodine a poziadas ich o pomoc, alebo to vydrz a po skonceni skoly si najdi pracu a byt. Jednoduche to nebude.

cacianka
•
Dnes o 16:47
@solange7 buď rodine alebo sociálke , ale Kajka nechce nič, iba odísť. A to nemôže, je maloletá....
Ani z voza ani na voz

mandarinka555
•
Dnes o 16:56
@kajka229 Rady od žien odmietaš...Máš milión výhovoriek, prečo to nemôžeš nikomu povedať...Takže čo čakáš? Že sa ti pozbierame na podnájom? Kto chce, hľadá spôsoby...Kto nechce, hľadá dôvody...

zuzanazu85
•
Dnes o 16:58
Je to trošku zvláštne,už v názve témy.... to oslovenie "mamina". Je to naozaj tak,že pije a bije ťa,alebo sú v tom iné dôvody? Iné možnosti asi nemáš,buď povieš rodine,alebo určeným inštitúciám. Si neplnoletá.

kajka229
AUTOR
•
Dnes o 16:58Včera som bola na pohovore v súkromných jasliach. Robila by som osmičky každý deň. Majiteľka mi povedala, že okrem starostlivosti o deti by som upratovala, vynasala smeti, robila raňajky, desiatu, olovrant, vydávala obedy, umyvala riady, robila každý deň s deťmi aktivity. Keď sa ma opýtala koľko si predstavujem plat, povedala som 850-900 v čistom. Ona ze ci som normálna, že ponúka 685 v čistom. Ja som ju vysmiala a išla preč. Baby, myslite, že som prestrelila výšku platu? To vážne tam človek robí učiteľku, kuchárku, upratovačku a zarobí necelých 700? Je to trenčiansky kraj.


lisymbia
•
Dnes o 16:00
😀😀😀 nech to robí sama za tu almužnu

pribinak
•
Dnes o 16:01
By som sa zasmiala a šla preč

lenkasarka
•
Dnes o 16:02
Dobre si spravila.
AUTOR
•
Dnes o 16:04
@lisymbia
@pribinak
@lenkasarka takéto zdieranie a medzi rečou sa mi pochválila, že o mesiac odlieta na dovolenku, tak rýchlo zháňa zamestnankynu

30katrusa40
•
Dnes o 16:05
Tak ak to má byť plat za plný úväzok je to málo. U nás sú súkromné jasle a viem že baby tam robia denne len doobeda od 8-12 a potom je tam už poobede pri odovzdávaní len majiteľka , učí tam tiež, tak je plat ok . Vždy záleží od pracovnej doby

pribinak
•
Dnes o 16:05
Toto čo si spisala sú 2-3 pracovne miesta. Ucitelka, kuchárka a upratovačka
AUTOR
•
Dnes o 16:06
@30katrusa40 plný úväzok 8,5 hodiny. Od 7 do 15,30
AUTOR
•
Dnes o 16:07
@pribinak no ved to, že robiť jedlo okrem obedov, na olovrant variť kašu, puding. Keď deti spia, upratovať

30katrusa40
•
Dnes o 16:08
Tak to je ozaj malo. To by Ťa len zdierala. Však v štátnej by si snáď viac zarobila.
AUTOR
•
Dnes o 16:08
@30katrusa40 pôjdem podať žiadosť do štátnej MŠ
AUTOR
•
Dnes o 16:09
@30katrusa40 plus by som prevozila 40 eur

30katrusa40
•
Dnes o 16:12
Držím palce aby si dobre a rýchlo našla prácu.
AUTOR
•
Dnes o 16:12
@30katrusa40 ďakujem pekne

gemerka17
•
Dnes o 16:15
Autorka len pre upozornenie. Plat sa vždy udáva v hrubom. V čistom by si ty mohla mat 900 eur, ak máš doma vlastné 3-4 malé deti, zatiaľ čo reálne by ti ponúkala šéfka minimálnu mzdu.
AUTOR
•
Dnes o 16:16
@gemerka17 ona sa opýtala koľko chcem v čistom

gemerka17
•
Dnes o 16:22
Aj u nej finančná gramotnost a vseobecny prehľad nula bodov. Môžeš na tu otázku odpovedať, že 1200 v hrubom napr. Ona nemôže vedieť koľko máš deti, ktoré by ti zvýšili čistú mzdu alebo pripadne exekúci, ktoré ti znížia mzdu na výplatnej páske. Nauč sa aj ty používať mzdovu kalkulacku.
A áno u sukromnikov je to tak, že tie platy chcú dat čo najslabšie a človeka tam zodrať.
AUTOR
•
Dnes o 16:24
@gemerka17 ona vedela zo životopisu že mám dve deti na základnej
AUTOR
•
Dnes o 16:25
Aj sa pýtala počas pohovoru na ich vek.

vevapeto
•
Dnes o 16:25
Urobila by som to isté, nech si pani majiteľka hľadá iného hlupáka.

ysseba
•
Dnes o 16:29
Dobre si urobila, ze si ju s takou almuznou za 3 pracovne pozicie vysmiala 👏

betulinka1
•
Dnes o 16:30
Mna by zaujimalo ako to chodi, ked ucitelka v jasliach ma toto vsetko robit a tie deti ako zaviaže a da do ohrádky? Ved su deti co nespia, treba ich mat stale na ociach, venovat sa im, taketo male nemozu byt v miestnosti bez dozoru. Cele zle taketo jasle. Mala som maleho v štátnych jasliach a boli tam 2 ucitelky a 1 zdravotna sestra na 8 deti, Cize 3 dospelé osoby. Po tom ako dali deti spat, jedna odchádzala domov. Jedlo im nosili z vedľajšej skolky. Na prechadzke boli vzdy 3 dospelé osoby, niekedy aj riaditelka ale ta mala na starosti viac zariadení.

letfenixa
•
Dnes o 16:33
@betulinka1 súhlasím. Veď kým pripraví jedlo všetkým deťom, nemôžu byť samé.
AUTOR
•
Dnes o 16:36
@betulinka1 no vraj sú tam dve učiteľky. Jedna robí jedlo, druhá strazi. Jedna uspava, druhá upratuje.
AUTOR
•
Dnes o 16:37
@letfenixa vraj jedna naloží obed, umyje riady, druhá umyje a prebali deti. Je tam 14 deti.

kepassa
•
Dnes o 16:56
Mam kamosku v sukromnej ms kde su deti od 2 r a okrem varenia robia vsetko. Ma 880€ v cistom a to chce ist uz do statnej ms, ze to uz nedava treti rok po sebe...necudujem sa

lucyk1992
•
Dnes o 17:04
A boli to jasle na dedine, alebo v meste? Lebo jedny jasličky mi to pripomína 🙈

kiki68
•
Dnes o 17:13Nepočula som o ničom takom ako pokuta za prechod 🙈 ale zas s deťmi som veľmi opatrná na ceste nespolieham sa na vodičov ani na prechod pre chodcov iba sama na seba

ms_green
•
Dnes o 15:13
Pokutu asi dostať môže ale ak to nie je nikde natočené v živote jej to nedokážu. Ty povieš, že ťa ohrozila, ona že nie a je to slovo proti slovu.

matullienka
•
Dnes o 15:14
Možno keby tam stáli policajti tak ju zastavia a nejakú pokutu jej dajú ak by videli ako to bolo. Ak tam nie je kamera zbytočne budeš volať políciu aj keby si vedela ŠPZ bude to len tvrdenie proti tvrdeniu

zuzubb007
•
Dnes o 15:14
Nikdy nevstupujem na prechod, pokým sa 3 x nepozriem ci dačo ide, tu by mali byť pokutovaný aj tí čo vstúpia zrazu na prechod, nielen vodici

tundra
•
Dnes o 15:16
Nedanie prednosti je za 20€,ohrozenie chodca za 150€
https://soferuj.sk/informacie/pokuty

micusa2222
•
Dnes o 15:21
to by som nič iné nerobila, len volala policajtov...treba poukázať aj na druhú stranu, na chodcov. povedzme si, koľko je tak drzých a arogantných, že keď vidí autá , aj v 50-km rýchlosti, a just stúpi na vozovku,,,,a nehovoriac o tých "matkách", ktoré tlačia pred sebou kočík.......brrrr....

hipuska
•
Dnes o 15:21
Vieš co, ja nechápem ľudí, ktorí vstupujú na prechod bez toho, žeby nemohli bezpecne prejsť. Ešte zvlášť, keď máš pred sebou kocik. Týmto si nezostávam šoféra. Len si treba uvedomiť, že pri strete s autom ty riskujes život a nie on.
AUTOR
•
Dnes o 15:25
@zuzubb007 pozerala som sa aj 3x, a nikdy sa na to nespolieham ze tlacim kocik pred sebou, ale ked uz niekto z dialky vidi ze je osoba na 3 ciare tak myslim ze staci jednou nohou byt na prechode a vodic musi zastat
@hipuska ja som na ten prechod uz dalej nesla o tom je tato tema
AUTOR
•
Dnes o 15:26
@tundra diky za info

zuzubb007
•
Dnes o 15:27
No na to sa netreba spoliehať, to, že si na prechode nohou neznamená, že je chodec nesmrteľný, vidím auto radšej čakám kým mi nekyvne rukou, že môžem ísť alebo nevidím, že stojí auto, podľa mňa je chyba na obidvoch stranach, za mňa ako vodiča je správanie chodcov strašné čo vidím tým sa nezastavam vodičov aj tí sú kadiaki
AUTOR
•
Dnes o 15:28
@zuzubb007 bola som na 3 ciare ale nevadi

matullienka
•
Dnes o 15:29
Tak môžeš zavolať na políciu a požiadať ich aby vykonávali častejšie kontroly, že sa ti už párkrát stalo, že ťa auto ohrozilo na priechode pre chodcov

zuzanazu85
•
Dnes o 15:30
Prepáč,ale aj niektorí chodci cez prechod si myslia ,že sú opancierovaní a idú ako s klapkami na očiach. Aj 3-4x sa pozrieť, pretože jedno pustí a vedľa ďalší nie. Cez prechody vždy opatrne a predvídavo.

limoncino
•
Dnes o 15:33
@zuzubb007 presne, som vodič, aj chodec a kým auto nestojí, cez prechod neprejdem aj keby bol neviem ako ďaleko. Ako vodič idem ozaj pomaličky pred prechodom, lebo niektorí si ozaj myslia, že majú okolo seba nejaky štít nesmrteľnosti či čo. Raz mi tak vybehla poštárka sediaca na bicykli, aj som jej pohladkala koleso, našťastie som už takmer stala, milá pani nič.. a o trende kolobezkarov asi pomlcim

letfenixa
•
Dnes o 15:33
Ale veď tretia čiara je už stred prechodu v 1 pruhu, to je vyslovene ohrozenie života.

ms_green
•
Dnes o 15:34
Sa mi páčia tie komentáre, že vstupovať na prechod len keď nejde auto. To by som v BA nikdy cez žiadny prechod neprešla. 😅 Nehovorím skákať pod auto, to určite nie.

jana_eyre_2
•
Dnes o 15:35
@zuzanazu85 prepáč ale vyhlaska hovorí, že chodca na prechode má prednosť a sofer má povinnosť zastaviť ak už je chodca na prechode.. autorka píše že bola na 3tej čiare, čo je už dosť výrazne na ceste a tak bola vodička povinná dať prednosť.. byť na tretej čiare nie je vybehnutie do cesty..

Autor neviem ako dokázať čo sa stalo, jedine kamerou, ale ano ak by tam bola hliadka tak by dostala pokutu (resp.mohla by).. sama som šofér aj chodec a poznám obe strany, často aj chodcov púšťam (ako to býva úplne bežné v európskych krajinách).. ale ak už je niekto na prechode, tak ma jednoducho prednosť..

zuzanazu85
•
Dnes o 15:38
@jana_eyre_2 veď samozrejme,ja sa nezastavam šoférky,len konštatujem. Práve pre tieto prípady sú aj dobrý výmysel, zelené svetlá vpredu.

heni79
•
Dnes o 15:41
Jazykové okienko - je to priechod pre chodcov. 😉 A k téme, poznám situáciu z oboch strán. Ako chodec vždy stojím a čakám, kým auto zastane. Minimálne z mojej strany, protismer spravidla už reaguje tiež. No už som aj stála v strede vozovky, na ostrovčeku. Ako vodič sa snažím vždy zastať, ak vidím chodca na priechode (či tesne pred ním). No ak sa mi ktosi vrhne pred auto (napr. bežec včera ráno), dupnem na brzdy a nadávam mu. On ma samozrejme nepočuje. 😉 S kočíkom buď vždy mimoriadne opatrná. Prvé v rane je tvoje dieťa! Takže radšej počkaj na chodníku, kým auto zastane.

matullienka
•
Dnes o 15:41
Inak už som zažila ako šofér aj také, že ja som zastavia a šofér čo išiel 50 m za mnou skoro do mňa zo zadu nabúral a ešte mi vynadal či budem stáť na každom prechode ( stála som na prvom) , ľudia nechápali čo tam boli,tak potom som naozaj zastavila na každom je h ho aj šľak trafi debila keď nevie ako sa má chovať

heni79
•
Dnes o 15:44
@ms_green V Ba sú semafory na 80% - 90% priechodov. Čiže prejdeš. 👍

heni79
•
Dnes o 15:47
A ešte „milujem" jednu vec. Stoja dve osoby na chodníku pred/pri priechode pre chodcov. Zastanem s autom, oni nič. Rozprávajú sa. Ukážem gestom, že môžu ísť. A oni mi gestom ukážu, že mám ísť ja. 🤦 Prečo stoja a debatujú tak, že to vyzerá, že chcú prejsť cez cestu? 🤦🤷

april171
•
Dnes o 15:50
A dokedy má chodec teda čakať na vodiča či zastane? Ak je auto napríklad 50 m od priechodu, tak vy čakáte kým príde a zastane? A keď je 100 m tak tiež? Tu nejde o hádzanie sa vodičom do cesty, ale keď vyhodnotím, že auto má priestor zastaviť, tak vojdem na priechod a nečakám kým príde k priechodu a zastane. To sa na mňa väčšina vodičov akurát tak vyprdne, môžem čakať akurát tak na Godota.
Ale samozrejme, stále si kontrolujem autá. Aká krajina, takí vodiči.

ms_green
•
Dnes o 15:51
@heni79 Myslela som zrovna tie prechody, kde semafóry nie sú. 😅Ako periete v práčke handry? Na mop, na prach, atď. Púšťate práčku na 60 kvôli napr. 5 handrám alebo s čím to periete?


lenqua_12325
•
Dnes o 14:50
Periem na krátky program so Sanytolom.

ivana11155
•
Dnes o 14:51
Periem zvlast na kratky program a pustam kvoli tomu viac-menej prazdnu pracku.co uz

dadka1997
•
Dnes o 14:51
Normalny program + na konci sa naparuje ... klasika -davam prací gél a pridávam aj ocot

viestta
•
Dnes o 14:53
Upratovacie handry periem s kobercekom z kupelne, na 40 s normalnym gelom.

lisymbia
•
Dnes o 14:54
Na 60 stupnov a pridám aj rohožky aj kúpeľňové koberčeky, pripadne hračky pre psa

maca1984
•
Dnes o 14:54
Ja pustím práčku aj len kvôli jednej násadke na mop.
🤷‍♀️ treba to oprať, tak čo už.

la_dolce_vita
•
Dnes o 15:09
Dam 15 minutovy program a staci to, o chvilu zasa pouzivam.

dadis
•
Dnes o 15:10
😳🙈 Ja to hadzem do pracky s normalnym pradlom po kazdom pouziti. Neznasam smradlave a spinave handry a po jednom pouziti mi absolutne nevadi prat ich s ostatnym pradlom..

martinka13
•
Dnes o 16:03
Samostatne na 60 s pracim gélom

nevenka
•
Dnes o 16:33
No, kedze mna nudza naucila, ze voda je vzacna, neexiat, ze to dam prat, ze 5 malych handriciek. A ked vidim, kolko litrov si pracka zoberie vody...Takze dam prach, gel, handry do hrnca, zalejem vriacou vodou a takto vyvarim, pomiesam to vareskou, ked to je uz take, ze to znesu ruky, tak to preperiem rucne, vyplacham, dam susit. A co je moc spinave, leti do kosa. Casto pouzivam stare tricka, pyzama na handry.

mimikarol
•
Dnes o 16:38
Ja to periem s ostatným pradlom. Vždy periem na 60C a keď sa mi zdá že tam dám dáku moc špinavu handru tak pridám dezinfekciu na pradlo.

slavomira236
•
Dnes o 16:42
Na upratovanie zvycajne pouzivam stare tricka to hned vyhadzujem. Mop namocim do vedra do horucej vody zvycajne preperiem necham postat este. Take na lestenie okien kachliciek normalne do pracky hodim raz za cas ku pradnu.

matullienka
•
Dnes o 15:53
@april171 ja napríklad ak vyhodnotim, že stihnem prejsť tak nečakám, len je rozdiel keď vidím, že to auto ide normálnou rýchlostnou alebo letí ako blázon. A zo zásady cez priechod keď idem tak pridám do kroku alebo podbehnem aby aj stojace autá nečakali kým prejdem šuchtavym krokom

jana_eyre_2
•
Dnes o 15:57
@april171 mne sa strašne "ráta" že ak je na MK diskusia o šoférovani, vždy sa začne tak trocha chápať len strana šoférov, aj keď v príbehu nie sú podľa vyhlášky v práve.. (nejako tu všetky opomenuli tú tretiu čiaru prechodu..)

Aspoň vidno prečo je na cestách toľko problémov... Každý si vyhlášku vysvetľuje po svojom...

Mne raz zrazil skoro syna jeden blbec lebo ignoroval že sme v strede prechodu.. a nie, nevbehli sme do cesty, keď sme stupili na prechod nič nešlo.. on vychádzal z bočnej cesty a bol povinný zastaviť.. on dupol na plyn a ja som strhla syna k sebe.. ak by som to neurobila, je mŕtvy.. tak nech mi ešte niekto povie o tom, že ja mám ísť na prechod ak nič nejde..

Ja som šofér a pravidlá sa snažim čo najviac dodržovať..

belllinka
•
Dnes o 16:01
@april171 ale nepreháňaj, vždy zastanú, nikdy nečakám dlhšie ako pol minuty

Nie, ved ja som mala v roku 2016 nejako 850/950 cisteho.
AUTOR
•
Dnes o 17:39
@kepassa majiteľka mi na pohovore povedala ze je to fyzicky aj psychicky náročná práca. Len keby to ocenila platovo.
AUTOR
•
Dnes o 17:39
@lucyk1992 v meste
AUTOR
•
Dnes o 17:40
@kepassa tu sú deti od roka do 3
@mandarinka555 nie nečakám od vás že sa mi vizbierate

margarita
•
Dnes o 17:14
až by si "ušla" z domu, tak či tak Ťa bude hľadať polícia so Sociálnou kuratelou a tým skôr tipujem, že v tom decáku môžeš skončiť.

Odporúčam Ti skúsiť kontaktovať Informačnú kanceláriu pre obete trestných činov v Tebe najbližšom okresnom meste -
https://www.minv.sk/?kontakty-2-1

Ak sa cítite byť obeťou, sú Vám BEZPLATNE A DISKRÉTNE poskytnuté základné informácie a usmernenie a sprostredkovaná odborná pomoc v oblasti:

psychologického poradenstva,
sociálneho poradenstva,
právnej podpory a usmernenia.

Mám skúsenosti s Ba pobočkou, vedia vypočuť, usmerniť, poslať na ďalšie inštitúcie, majú stovky reálnych skúseností z praxe.
Riešia čokoľvek od týraných žien cez oklamaných seniorov, kyberšikanu, diskrimináciu v práci alebo znásilnené dievčatá, čo sa hanbia/boja/nechú ísť na políciu hneď...

Držím palce, rozhodne zájdi a zdôver sa aspoň niekomu....
Dnes o 17:14
@deti95060810 ja mam obvodného,ktorý mi nechce predpisovať ten liek Diazepam 5mg a mal bi však???

viestta
•
Dnes o 17:37
Je mi luto tvojej straty, ale nemas niekoho s kym by si zdielala svoj smutok? Vyrozpravat sa, vyplakat, nechat svoj smutok plynut a prijat, ze sa to uz nezmeni. Niekto potrebuje butlavu vrbu, niekto chce byt v tazkych chvilach sam. Ty vies co by si potrebovala. Toto ti pomoze viac ako lieky, aj ked netvrdim, ze nie su chvile ked su aj lieky na mieste
AUTOR
•
Dnes o 17:43
@viestta zdieľame tento smútok, šok viaceré kamošky.
Samozrejme.
Voláme si, píšeme si cez messenger.
Manžel mi je veľmi nápomocný, aj o pol 2 v noci počúva môj plač, reči.
Ale sama cítim, viem, že toto nie som ja.
A neviem fungovať v " normálnom stave".
Som unavená, nevyspaná.
Hľadám odpovede..
Ja viem, že to prejde.
Len je to veľmi čerstvé, veľmi ma to šokovalo.
•
Dnes o 17:36
@nika11103 to poznám ja si dám len cez víkend pretože mám hroznú prácu v ktorej som nešťastná a keď je víkend konečne som rada že si dám pohár vína a idem spať 🤣 takže nieste sama.... Mne nevadí sa vzdať sladkosti a ani vôd sladkých ja budem len rada 🙏🏾 len som aj dosť závodnena a naozaj potrebujem schudnúť do 2 mesiacov najmenej 8 kg aj keby to malo byť s plačom a drasticky a potom sa len udržiavať

lucia13022023
•
Dnes o 17:42
A za ten čas co si nejedla pečivo a sladké si schudla? Nejesť niečo neznamená ze schudneš, schudneš len keď budeš v deficite. A v deficite môžeš jest aj to pečivo. Skus napísať tvoj denný jedálniček, plus pohyb aký máš a skúsim ti pomoct 🙂

lisymbia
•
Dnes o 17:42
Tak v prvom rade, pri chudnutí je viac dôležitejšia strava ako pohyb. Za ďalšie, to prečo sa ti to nedarí je pravdepodobne to že ideš na 100% čo bohužiaľ nie je šanca dodržať. Musíš si udržovať balanc teda 80/20. Teda 80% vyváženej stravy so správnymi makrami, výživovou hodnotou. Zvyšok tých 20% tvorí všetko ostatné čo máš rada či už čokoláda, chipsy a podobne.

amelia89
•
Dnes o 17:43
@viktoriaviktoria11 mam 168cm a v januari som mala 73kg. Jem podla kalorickych a kazdy den mam nieco sladke - kolac,zmrzlinu atd. Ked aa to naucis mozes jest uplne vsetko len si strazis to mnozstvo, proste nemozes mat aj kolu aj kolac aj zmrzlinu aj vsetko jedlo v 1 den…
no kazdopadne cvicim,jem rozumne a mam dole 5kg ale necitim ziadne obmedzenia. len trochu sa naucit s tym pracovat a premylsat nad tym. to co robis ty nie je nikdy dlhodobo udrzatelne to telo si to vypyta proste
„Ale, Vojto, čo ťa nemá. To ten dnešný kúzelný deň,“ smial sa starý otec. Ale Vojtík vedel. O to väčšiu radosť z toho mal.

Vojtík s dedom prišli z prechádzky domov. Keď ich babka uvidela, spraskla ruky.

„Kde si sa tak zašpinil, Vojtík?“

„Hľadal som poklady! A našiel, pozeraj,“ povedal Vojtík a ukazoval babke poklad.

„Tie nohavice ti vyperiem, ale až zajtra. Dnes sa nesmie prať.“

„Prečo?“ pýtal sa Vojto udivene. Zrazu si spomenul, že pre samé poklady úplne zabudol na príbeh o Ježišovi.

„Pretože dnes je veľmi smutný deň. Dnes bol Ježiš odsúdený a ukrižovaný,“ vysvetlil starý otec. „Hovorí sa, že kto by dnes pral bielizeň, tak by pral v jeho krvi.“

„On teda naozaj zomrel?“

„Áno, Vojtík. Jeho žiaci mu urobili hrob a zavalili ho veľkým kameňom.“

„Takto to teda končí?“ pýtal sa smutne Vojtík.

„Kdeže. To bude ešte pokračovať. Rozprávam ti predsa rozprávku a tie majú dobré konce, nie? Tak utekaj domov. Zajtra si pripravíme korbáč.“
„To viem. To je ten panáčik, čo visí tamto v obývačke na kríži.“

„Bol to človek, ktorý tu podľa príbehov žil pred dávnymi a dávnymi časmi. Bol synom Božím a prišiel na svet, aby ľudí naučil, ako sa k sebe správne správať. Učil ľudí, že sa majú mať radi, nesúperiť spolu, mať s druhými zľutovanie a vzájomne si pomáhať. Zatiaľ čo na Vianoce ľudia oslavujú deň, kedy sa narodil, na Veľkú noc si pripomínajú deň, kedy zomrel a zase obživol.“

„Jaj. A čo sa mu stalo?“

„Našli sa takí, ktorým sa jeho učenie o ľudskej dobrote nepáčilo. Možno mu aj závideli, že ho majú ľudia radi, a chceli sa ho zbaviť. Jeden z Ježišových žiakov, Judáš sa volal, Ježiša zradil. V ten deň sa Judáš na Ježiša celý deň mračil. Preto sa o tomto dni hovorí, že je to Škaredá streda.“

„A ako ho Judáš zradil?“

„Povedal ľuďom, čo sa chceli Ježiša zbaviť, kde ho nájdu.“

„Prečo to urobil? On ho nemal rád?“ čudoval sa Vojtík.

„Nechal sa zlákať peniazmi. Za tridsať strieborných prezradil, kde na druhý deň Ježiša nájdu.“

„A Ježiša chytili?“ pýtal sa Vojtík dychtivo.

Starý otec sa záhadne usmial.

„Ako to bolo ďalej, ti poviem zajtra. Teraz poď, pôjdeme babičke pomôcť piecť judáše. To je to sladké pečivo, čo máš rád.“
Vyšiel von pred dom, a poobzeral sa po okolí. No psíka nevidel..no jeho oči padli na niečo sivé. Natrafil na malú sivú mačku.

Rozprávka na čítanie - Hrdina Hafík
Hrdina Hafík
„Ahoj nevidela si takého hnedého malého psíka? Je huňatý a volá sa Hafík, a že vraj utekal preč..“ spýtal sa chlapček pouličnej sivej mačky. Tá sa na neho potmehúcky pozrela, oblízala si labku a zamňaukala: „Psíka..tých nemám rada,“ povedala, „ale áno bol tu jeden..,a utekal za nejakým dievčaťom!“ povedala nakoniec a odišla preč.

Chlapček sa jej poďakoval a hľadal ďalej. Prešiel zopár metrov, keď v tom začul štekot. A nie hocijaký štekot! To bol predsa Hafík!  To on štekal! Chlapček nasledoval zvuk a utekal čo mu len nohy stačili. V tom uvidel jeho susedku, dievčatko menom Zorka, ktorá sa práve hrala s Hafíkom.

„Hafík, čo tu robíš?“ povedal vyplašene Martinko a začal objímať svojho psíka. Hafík sa mu samozrejme tešil tiež a celého ho vyoblizoval.

„Tvoj psík ma zachránil! Je to hrdina““ povedalo mu dievčatko. Martinko sa neveriacky pozrel na svojho psa a potom na Zorku. Jeho pes a hrdina?

„Hrala som sa na ulici, kreslila som si kriedou po chodníku, keď v tom sa vyrútil veľký pouličný pes. Veľmi som sa bála a kričala o pomoc.  A práve Hafík ma zachránil! Pribehol a svojím štekotom psa odohnal! A teraz tu na mňa dáva pozor ako si kreslím na chodníku!“ dopovedalo príbeh dievčatko. Martinko bol na svojho psíka pyšný. Chvíľku sa ešte hrali a potom sa detičky pobrali domov. Rodičia a aj Martinko sa tešili, že je ich psík konečne doma s nimi.

Psík Hafík síce vedel, že mal čakať Martinka, no keďže je superhrdina musel dievčatku pomôcť! Teda aspoň si myslel, že je superhrdina.. i keď vlastne aj jedným bol. Pomohol predsa dievčatku a nie len to. S Martinka a Zorky sa stali kamoši a všetci traja sa hrali spolu na superhrdinov. Martinko vedel, že by sa Hafík nakoniec k nemu vrátil, no bol rád, že ho našiel.. je to predsa odvážny hrdina!
„Ahoj, nevidel si Hafíka?“ spýtal sa chlapček pavúka, ktorý si tam práve priadol pavučinu. Pavúk sa zamyslel. 

„Veru tu som ho nevidel. Dnes som ho videl len na dvore.. a potom za niečím utekal!“ spomenul si nakoniec pavúčik. Martinko poďakoval pavúčikovi za túto informáciu a vybral sa do záhrady. 

„Hafík, kde si? Hafik!” kričal smutne ako behal po záhrade. V búde nebol, ani za domom..hmm kde len je. Toto sa na Hafíka nepodobá.. vždy ho predsa verne čakal pri bráničke.. aby sa potom spolu hrali.  „Jój..čo keď sa mu niečo stalo! Pavúčik hovoril, že niekam utekal!” rozmýšľal Martinko nahlas. Určite sa mu len niekde schoval. Pozrel hore na strom, kde sedela mala ryšavá veverička. Čo keď bežal Hafík za ňou..musí sa jej spýtať! 

„ Ahoj, nevidela si môjho psíka Hafika?” spýtal sa jej Martinko zvedavo . No veverička len nechápavo pokrútila hlavou, že žiadneho psíka nevidela. Martinko sa nevzdával a hľadal ďalej. Jeho rodičia zatiaľ chodili po ulici a tiež hľadali Hafika. Chlapček obehol záhradu, kde pri kriku malín natrafil na ďalšie zvieratko, ktoré možno Hafíka videlo. V malinovom kríku, na jednom lístku sedela malá včielka. 

„Ahoj pani včielka, nevidela si môjho psíka Hafika?” spýtal sa Martinko. Včielka priletela bližšie k chlapčekovi a zabzučala: „Hmm..ale áno, utekal preč!“ povedala chlapčekovi nakoniec a zatrepotala krídelkami.  

„To mi povedal aj pán pavúčik…a nevieš kam utekal?“ spýtal sa jej Maťko. 

„Utekal von na ulicu, počul kričať malé dievčatko!“ spomenula si nakoniec pani včielka a odletela preč. Martinko jej vrúcne poďakoval a vybral sa na ulicu. Žeby jeho odvážny psík išiel niekomu zachrániť život? Často sa predsa hrajú na superhrdinovi. Vyšiel preto von a hľadal ho po okolí. Vedel, že sa nájde, je predsa veľmi odvážny.  A to ako táto rozprávka skončí sa dozvieme v ďalšej časti.

Rozprávka na čítanie - Jazvece Monty a Piškót
Jazvece Monty a Piškót
Po chvíli hľadania Monty zrazu zakričal: „Piškót, počul si to?“ Nie, nič som nepočul.“ Piškót sa naňho nechápavo pozrel. Monty ukázal labkou na ústa a povedal Piškótovi, že má byť ticho a počúvať. Niekoľko sekúnd sa nič nedialo, ale potom to obaja počuli. „Pomóc, pomôžte nám niekto! Prosím!“ ozvalo sa odniekiaľ z diaľky.

Jazvečí chlapci okamžite spoznali, že ich volajú veveričie kamarátky. Neváhali a rozbehli sa za ich hlasom. Preskočili čučoriedkové kríky, odhrnuli ich a vtedy to uvideli. Veveričky spadli do obrovskej jamy, ktorá bola v lese vykopaná. Dieru vykopali pytliaci, aby chytili nejaké zviera. Teraz však do nej spadli dve veveričky a nemohli sa dostať von. Jazvece sa k nim okamžite rozbehli a rozmýšľali, ako veveričky dostať von. Diera bola hlboká a po stranách sa nedalo ničoho chytiť.

Až potom Piškótovi niečo napadlo: „Monty, nájdeme dlhý konár, priviažeme ťa na jeho koniec pevnou trávou a ja ťa spustím dole k veveričkám. Ty im podáš ruku a oni vylezú hore po tebe a po konári.“ 

 Obaja začali hľadať najdlhší konár a najsilnejšiu trávu, ktorú zplietli do lana. Monty si priviazal nohu ku konáru, ale keď sa chcel spustiť dolu, dostal strach. Stál nad dierou, pozeral na uväznené veveričky a úplne ztuhol.

„Piškót, ja sa bojím,“ smutne pozrel na svojho bračeka. „Monty, ja ti verím. Viem, že tá diera je hlboká, ale tiež viem, že to dokážeš. Neboj sa. Strach je normálny, ale cez strach prichádza odvaha,“ snažil sa Piškót Montyho upokojiť.

A tak sa Monty odhodlal. Pozrel sa dolu, zhlboka sa nadýchol a pomaly sa spustil takmer na dno jamy. Pomohol svojim veveričím kamarátkam dostať sa na vrch a potom všetci pomohli vytiahnuť Montyho hore. Spoločne jamu zasypali, aby do nej už nikto nikdy nespadol, a vybrali sa domov.

Keď neskôr večer jazvecí chlapci všetko rozprávali rodičom, tí neverili vlastným ušiam. Boli na seba hrdí. Odvtedy Monty nikdy nezabudol, že strach je normálny a že sa zaň nemusí hanbiť. A tiež vedel, že cez strach prichádza odvaha.
„Medík je liek! Určite ti pomôže!“ zvolal. Medvedík bol presvedčený, že jeho medík je silná medicína a že vtáčikovi skutočne pomôže.

A tak aj bolo. Medvedíkov medík skutočne pomohol na ranené krídlo. A to hlavne preto, že tomu obe zvieratká verili!
:)referát patrí k žánrom náučného štýlu. Využíva sa informačný slohový postup,opsiný a výkladový. Informuje o výsledkoch vedeckého výskumu alebo vysvetľuje dosiahnuté vedecké riešenia(postupy).

POSTUPY PRI PRÍPRAVE REFERÁTOV:
1.Téma:

2.Zdroje-zháňame čo najviac informácií z literatúry(knihy,internet, ..) a vypíšeme si kľúčové slová.:

3.Osnova-Zostavíme osnovu(úvod,jadro,záver).:

4.Referát-napíšeme referát

Úvod:
stručný,vysvetlíme v ňom prečo sme si vybrali určenú tému, alebo načo sa v referáte zameriame.

Jadro:
Hlavná časť,jednotlivé tematické celky sú samostatné, ale majú na seba logicky nadväzovať.Využívame viac odsekov, odborných slov.

Záver:
Stručný, zhrnieme výsledky výskumualebo zdoraznime význam prednesenej témy.

REFERÁT je vopred písane pripravený ale prednáša sa ústne.

Zásady pri ústnom prejave:
1. Hovorit primerane nahlas                                              
Nárečie alebo dialekt je územne a funkčne vymedzený štruktúrny jazykový útvar, ktorým spontánne komunikuje autochtónne obyvateľstvo istej oblasti.Je to územný (zemepisný) variant národného jazyka s vlastným zvukovým, gramatickým, slovotvorným a lexikálnym systémom (napr. stredoslovenské nárečie, východoslovenské nárečie, západoslovenské nárečie), tvoriaci komplexnú lingvistickú, historickú a sociologickú kategóriu. Metodické preferovanie jednotlivých znakov týchto kategórií sa odráža v rozličných definíciách a hodnoteniach nárečia. Pri formulovaní téz o funkčne vymedzenej platnosti nárečia sa uplatňuje najmä sociologický aspekt.

Miestne (územné, oblastné, teritoriálne) nárečie má hovorený charakter; v rečových prejavoch prevažuje dialóg. V súčasnosti nárečia neplnia všetky sociálne funkcie, ktoré náležia plnohodnotnému jazykovému útvaru. Zo štyroch základných jazykovo-komunikačných činností (reč, písanie, počúvanie, hovorenie) sa v nich primárne uplatňuje iba prvá a tretia. Kognitívne funkcie definitívne prevzal jazykový útvar s vyššou prestížou, a to spisovný jazyk; zanedbateľné nie sú ani vplyvy jeho neštruktúrnych foriem.

Slovenské nárečia predstavujú dorozumievací prostriedok autochtónneho obyvateľstva príslušných nárečových oblastí v každodennom spoločenskom a pracovnom styku s najbližším okolím. Slovenské nárečia sa doteraz dedia z generácie na generáciu v ústnej podobe, hoci aj tu dochádza v porovnaní s minulosťou k procesu nivelizácie. Slovenské nárečia sa členia na tri základné skupiny:

Územné členenie slovenských nárečí
a) Západoslovenské nárečia

Západoslovenské nárečia sú rozšírené v trenčianskej, nitrianskej, trnavskej, myjavskej oblasti a v ďalších regiónoch.

Hornotrenčianske nárečia
Dolnotrenčianske nárečie Považské nárečia
Stredonitrianske nárečia
Dolnonitrianske nárečia
Nárečia trnavského okolia
Záhorské nárečie

b) Stredoslovenské nárečia

Stredoslovenskými nárečiami sa hovorí v regiónoch Liptov, Orava, Turiec, Tekov, Hont, Novohrad, Gemer a vo zvolenskej oblasti.

Liptovské nárečia
Oravské nárečia
Turčianske nárečie
Hornonitrianske nárečia
Zvolenské nárečia
Tekovské nárečia
Hontianske nárečie
Novohradské nárečia
Gemerské nárečia
c) Východoslovenské nárečia

Východoslovenské nárečia možno nájsť v regiónoch Spiš, Šariš, Zemplín a Abov.

Spišské nárečia
Abovské nárečia
Šarišské nárečia
Zemplínske nárečie
Národný jazykje diasystém tvorený viacerými formami, resp. štruktúrnymi útvarmi istého jazyka, používaný v danom národnom spoločenstve.

Sociálno-historickou bázou národného jazyka je národné spoločenstvo. Národný jazyk je spolu s ďalšími znakmi spravidla výrazným znakom národa.

Jazyka sa členení na variety (útvary, formy).

Spisovný jazyk (z nem. Schriftsprache) alebo štandardný jazyk (z ang. standard language) alebo skrátene štandard alebo literárny jazyk (z rus. literaturnyj jazyk resp. fr. langue littéraire) môže byť:

„najvyššia“, „prestížna“ celonárodná varieta/forma jazyka, synonymum: štandardná varieta, pozri štandardná varieta (prestížna forma)
jazyk, ktorý má (aj) jednu alebo viacero štandardizovaných, normovaných podôb, teda jazyk, ktorý má jednu alebo viacero štandardných variet, pozri spisovný jazyk (jazyk so štandardnou varietou)
 Spisovný jazykje v delení národného jazyka podľa Horeckého (a podobných deleniach) na spisovnú varietu (celonárodnú spisovnú formu) – štandardnú varietu (štandardnú spisovnú formu) –  subštandardnú varietu (subštandardnú spisovnú formu) [- nadnárečovú varietu (nadnárečovú formu)] – nárečovú varietu (nárečovú formu) [- a ako špecifickú skupinu jazyk umeleckej literatúry]:

a) spisovná varieta (t. j. spisovný jazyk ako najprestížnejšia forma jazyka)
b) spisovná varieta + štandardná varieta (hovorový štýl) + subštandardná varieta (t. j. spisovný jazyk ako opak nárečí)
c) spisovná varieta + štandardná varieta (hovorový štýl) (napr. v KSSJ)
Spisovný jazyk alebo literárny jazyk je zriedkavo: písaný jazyk.

Literárny jazyk je jazyk umeleckej literatúry

Veta je základná syntaktická jednotka s uceleným významom, gramaticky usporiadaná a intonačne uzavretá. Presnejšie je to komplexná gramaticko-sémantická systémová jednotka, ktorá má povahu základnej komunikatívnej jednotky.

Veta sa skladá z vetných členov, ktoré v nej vytvárajú sklady.

Vyjadrením postoja hovoriaceho ku skutočnosti, teda podľa obsahu (modálnosti), delíme vety na:

Oznamovacia veta: Obsahuje oznam. Má klesavú melódiu. Na konci píšeme bodku; pr. Pôjdeme spolu do školy.
Opytovacia veta: Obsahuje otázku, na niečo sa pýtame, niečo zisťujeme. Má stúpavú melódiu. Na konci píšeme otáznik; napr. Pôjdeš so mnou dnes do školy?
Rozkazovacia veta: Vyjadruje rozkaz. Má klesavú melódiu. Na konci píšeme výkričník. napr. Choďte dnes spolu do školy!
Želacia veta: Vyjadruje želanie. Má stúpavo-klesavú melódiu. Na konci píšeme výkričník. pr. Poďme dnes do školy spolu!
Zvolacia veta: Vyjadruje citové pohnutie, emóciu (strach, radosť atď.). Má stúpavú alebo stúpavo-klesavú melódiu. Na konci píšeme výkričník. napr. Tak sa mi to páči!
ŽELACIE VETY
Želacie vety sú prejavom vôľovej a citovej stránky psychických javov. Podávateľ nimi vyjadruje svoju vôľu, želá si, aby sa niečo stalo, alebo nestalo. Svoje želanie však nevyjadruje kategoricky, ale rozhodnutie o vyplnení vyslovenej žiadosti ponecháva prijímateľovi: Maj sa dobre! — Keby to boh tak!

Štandardný jazyk je všeobecne: štandardne používaný či nastavený jazyk (aj programovací a podobne)

Kultúrny jazyk je významná súčasť kultúry, kultivovaná reč istej spoločnosti. Nie každý jazyk je zložkou kultúry, ale iba kultúrny jazyk. Jazyk je nanajvýš jedným z predpokladov kultúry, no nie je automaticky jej zložkou. Zložkou kultúry sa stáva až pestovateľským úsilím členov jazykového spoločenstva, povedzme v súvislosti rôznymi obradmi, vedením štátu, zahraničnej diplomacie, rozvíjaním literárnych štýlov atď.

Nárečie alebo dialekt je územne a funkčne vymedzený štruktúrny jazykový útvar, ktorým spontánne komunikuje autochtónne obyvateľstvo istej oblasti.Je to územný (zemepisný) variant národného jazyka s vlastným zvukovým, gramatickým, slovotvorným a lexikálnym systémom (napr. stredoslovenské nárečie, východoslovenské nárečie, západoslovenské nárečie), tvoriaci komplexnú lingvistickú, historickú a sociologickú kategóriu. Metodické preferovanie jednotlivých znakov týchto kategórií sa odráža v rozličných definíciách a hodnoteniach nárečia. Pri formulovaní téz o funkčne vymedzenej platnosti nárečia sa uplatňuje najmä sociologický aspekt.

Miestne (územné, oblastné, teritoriálne) nárečie má hovorený charakter; v rečových prejavoch prevažuje dialóg. V súčasnosti nárečia neplnia všetky sociálne funkcie, ktoré náležia plnohodnotnému jazykovému útvaru. Zo štyroch základných jazykovo-komunikačných činností (reč, písanie, počúvanie, hovorenie) sa v nich primárne uplatňuje iba prvá a tretia. Kognitívne funkcie definitívne prevzal jazykový útvar s vyššou prestížou, a to spisovný jazyk; zanedbateľné nie sú ani vplyvy jeho neštruktúrnych foriem.

Sotácke nárečia
Užské nárečia
Oblasť goralských nárečí
Oblasť ukrajinských nárečí
Nárečovo rôznorodé oblasti
Oblasť maďarských nárečí
Tieto skupiny sa ďalej bohato a pestro členia („Čo dedina, to reč iná“), pričom členitosťou sa nárečia vyznačujú predovšetkým v hornatých oblastiach. Práve hornatosť krajiny spôsobovala v minulosti istú (rečovú) izolovanosť obyvateľstva v rámci jednotlivých žúp. Pod tieto špecifiká sa podpísalo ďalej aj prevrstvovanie a migrácia obyvateľstva, kolonizácie, miešanie odlišných nárečových typov, pôsobenie susedných slovanských i neslovanských jazykov, zmeny v zamestnaní obyvateľstva a pod.

Podľa povahy nárečí a výskytu jednotlivých charakteristických javov možno zaradiť do uvedených skupín aj slovenské nárečia v Maďarsku, Srbsku, Chorvátsku, Rumunsku, Bulharsku a v iných krajinách, kam sa v minulosti presídlili veľké kompaktné skupiny. Pri menšom počte starých písomných pamiatok sú slovenské nárečia základným prameňom slovenskej historickej gramatiky.
2. Spisovná výslovnosť                                             

3.Využívame doraz a prestávky v reči                                             

4.Nadviazanie očného kontaktu s poslucháčmi (nečítame všetko z papiera).                                             

5.Stáť uvoľnene.                                              

6. Využívame gestikuláciu a mimiku

Referát možeme doplniť  o obrazový a zvukový materiál.(obrázky,grafy,...).
Lomka (/) je interpunkčné znamienko.
Píše sa:

v odborných textoch medzi variantnými a protikladnými výrazmi , napríklad príbuzné/nepríbuzné jazyky (príbuzné alebo nepríbuzné jazyky, príbuzné a nepríbuzné jazyky), kategória odcudziteľnosti/neodcudziteľnosti, -mi/-ami (pádová prípona so základnou podobou -mi a s variantom -ami), -ár/-ar/-iar (slovotvorná prípona so základnou podobou -ár a variantmi -ar, -iar, umyť/umývať (sloveso s dokonavou podobou umyť a nedokonavou podobou umývať); Variantný výraz sa zvyčajne uvádza po spojke alebo sa uvádza v zátvorkách, napríklad: pribuzné a nepríbuzné jazyky, príbuzné alebo nepríbuzné jazyky, príbuzné (nepríbuzné) jazyky; pádová prípona -mi a jej variant -ami, pádová prípona –mi (-ami).
na vyjadrenie podielového alebo pomerového vzťahu dvoch veličín, napríklad 1/3, 2/5 (jedna tretina, dve pätiny); spotreba 6l/100km (šesť litrov na sto kilometrov); Rýchlosť hviezdy je 120 000 km/s (120 000 kilometrov za sekundu);
na zaznačenie školského roka, napríklad 1999/2000, 2000/01;
na oddelenie veršov v súvislom texte (tu sa pred lomkou aj za ňou vynecháva medzera), napríklad: V diaľke sa trasie. Vysoká. /Číhajú na ňu. – Byť samým sebou je byť / najmä keď si v množstve / plnom teplých ľudských dychov.
Lomka sa používa aj ako diakritické znamienko, ktorým je prekrížené písmeno, napríklad písmeno v nórskej alebo dánskej abecede.

Pri lomke — sa vo východiskových textoch používa lomka medzi spojkami a, alebo, čo sa ukazuje ako problematické. Všimnime si niektoré príklady: Vyvinúť a/alebo podľa potreby posilniť v spolupráci s príslušnými orgánmi..., ako aj združeniami/sieťami miestnych orgánov globálnu a ľahko dostupnú informačnú sieť.

V tomto príklade je písanie lomky namieste medzi slovami združenie, sieť, teda vo výraze združeniami/sieťami. Ten sa dá jednoducho rozpísať bez lomky a transformovať napríklad takto: ...posilniť v spolupráci s príslušnými orgánmi, ako aj združeniami a sieťami (resp. združeniami alebo sieťami). Výraz združeniami/sieťami vyhovuje pravidlu o tom, že lomka sa píše v odborných textoch medzi variantnými výrazmi (pozri Pravidlá slovenského pravopisu, s. 126 bod l).

Za sporné však pokladáme písanie lomky v citovanom príklade medzi spojkami a, alebo. Všimnime si ešte ďalšie príklady: Podporovať súkromný sektor tak, aby sa zlepšili a/alebo vytvorili finančné toky. — Posilňovať a/alebo rozvíjať globálne systémy včasnej výstrahy, aby obyvateľstvo bolo pripravené... — ...miestne orgány a/alebo občianske organizácie uviesť do činnosti... — ...ustanoviť a/alebo posilniť mechanizmus spolupráce... — ...ustanoviť a/alebo posilniť partnerstvo s medzinárodnými združeniami.
Melódia je tónové vlnenie slabík vetného úseku, spôsobované zmenou výšky hlasu po sebe nasledujúcich nositeľov slabičnosti. Melódia je gramatizujúcim prostriedkom vety. Sekundárne sa môže využiť ako expresívny prvok. Výkyvy vo výške tónu môžu mať aj štylizujúci účinok. Melódia vety je takto funkčne najzaťaženejším prozodickým javom v spisovnej slovenčine. Komunikačne je najcitlivejšia a najdôležitejšia melódia koncového vetného úseku, teda melódia predpauzových slabík.

V podstate rozoznávame tri druhy melódie:
1. konkluzívnu kadenciu (melódia vetného úseku uspokojujúco uzavretého končiacou pauzou). Je to predovšetkým melódia oznamovacích viet. Posledný vetný úsek má výrazne klesavý tónový priebeh; tón poslednej slabiky je najnižší.

2. antikadenciu (melódia vetného úseku neuspokojujúco uzavreteno končiacou pauzou). Ide o melódiu zisťovacích opytovacích viet. Má stúpavý alebo stúpavo klesavý priebeh.

3. semikadenciu čiže polokadenciu (melódia vetného úseku pred nekončiacou pauzou). Táto melódia naznačuje, že výpoveď sa ešte neskončila, počúvajúci očakáva jej pokračovanie. Slabiky predpauzového úseku majú dosť monotónny priebeh; tón poslednej slabiky tohto úseku nie je najnižší.

Prídavne mená sú plnovýznamový, ohybný slovný druh, ktorý pomenúva vlastnosti osôb, veci,

predmetov, javov. V slovenčine ich značíme číslom 2

Rozdelenie prídavných mien:

A)      Akostné prídavné mená: vyjadrujú určitú vlastnosť podstatného mena. Tento druh prídavných mien sa dá stupňovať, napríklad dobrý- lepší- najlepší, zlý- horší- najhorší

B)      Vzťahové prídavné mená: vyjadrujú vzťah k určitému podstatnému menu, napr. Štúr- štúrovský, Piešťany- piešťanský, Nemecko- nemecký

C)      Vzťahové živočíšne prídavné mená: vyjadrujú vzťah k určitému zvieraťu, napr. páví, sloní, jelení, medvedí

D)      Privlastňovacie prídavné mená: tvoríme ich od zvieracích a osobných podstatných mien   

Gramatické kategórie prídavných mien:

Rod: mužský (ten), ženský (tá), stredný (to)

Číslo: jednotné číslo (singulár), množné číslo (plurál)

Pád: Nominatív (N), Genitív (G), Datív (D), Akuzatív (A), Lokál (L)  a Inštrumentál (I)

Pádové otázky: Kto? Čo?, (bez) Koho?, Čoho?, (dám) Komu? Čomu?, (vidím) Koho? Čo?, (o) Kom? Čom, (s) Kým?, Čím?

Vzor: pekný, cudzí, matkin, otcov a páví

Poznámka: Životnosť pri prídavných menách sa neurčuje

Charakteristické znaky skloňovacích vzorov prídavných mien

Skloňovanie podľa vzoru pekný: Podľa vzoru pekný sa skloňujú všetky prídavné mená, ktoré pred dlhou  samohláskou ý majú tvrdú spoluhlásku.

Skloňovanie podľa vzoru cudzí: Podľa vzoru cudzí sa skloňujú všetky prídavné mená, ktoré pred dlhou samohláskou í majú mäkkú spoluhlásku.

Skloňovanie podľa vzoru matkin: Podľa vzoru matkin sa skloňujú všetky prídavné mená, ktoré sú v nominatíve singuláru zakončené na in.

Skloňovanie podľa vzoru otcov:  Podľa vzoru otcov sa skloňujú všetky prídavné mená, ktoré sú v nominatíve singuláru zakončené na ov.

Skloňovanie podľa vzoru páví: Podľa vzoru páví sa skloňujú všetky zvieracie prídavné mená, ktoré sú v nominatíve singuláru zakončené na í. V tomto vzore je porušené pravidlo o rytmickom krátení.

Poznámka: Pravidlo o rytmickom krátení hovorí, že v slovenčine nemôžu ísť po sebe 2 dlhé slabiky.

Vyskloňovanie vzorov prídavných mien:

Vzor pekný

Singulár - N: pekný, pekná , pekné, G:  bez pekného, bez peknej, bez pekného, D: dám peknému, dám peknej, dám peknému, A: vidím pekného, vidím peknú, vidím pekné, L: o peknom, o peknej, o peknom, I: s pekným, s peknou, s pekným

Plurál - N: pekní, pekné, pekné, G: bez pekných, D: dám  pekným, A: vidím pekné, L: o pekných, I: s peknými         

Vzor cudzí

Singulár - N: cudzí, cudzia, cudzie, G: bez cudzieho, bez cudzej, bez cudzieho, D: cudziemu, cudzej, cudziemu, A: vidím cudzieho, vidím cudziu, vidím cudzie, L: o cudzom, o cudzej, o cudzom, I: s cudzím, s cudzou, s cudzím

Plurál - N: dvaja cudzí, dvaja cudzie, dvaja cudzie, G: bez cudzích, D: dám cudzím, A: vidím cudzích, vidím cudzie, vidím cudzie, L: o cudzích, I: s cudzími

Vzor páví

Charakteristika podstatných mien

Podstatné mená sú ohybný plnovýznamový slovný druh, ktoré pomenúva osoby, veci, javy, predmety, duševné stavy. Pri značení slovných druhov ho značíme číslom 1

Rozdelenie podstatných mien:

A)     Abstraktné podstatné mená -  sú to podstatné mená, ktoré majú nehmatateľný charakter.

Príklady abstraktných podstatných mien: láska, šťastie, úsmev, krása, nenávisť, zlo, dobro.

B)      Konkrétne podstatné mená - sú to podstatné mená, ktoré majú hmatateľný charakter.

Príklady konkrétnych podstatných mien:  pero, papier, cukrík, noha, ponožka, tepláky, koberec, syr, šunka, vajce

C)      Životné podstatné mená - sú to podstatné mená mužského rodu , ktoré majú živý charakter. 

Pozor: Pri zvieracích podstatných menách mužského rodu sú životné len v jednotnom čísle a zvieracie podstatné mená mužského rodu v jednotnom čísle sa skloňujú podľa vzorov chlap a hrdina.

Príklady životných podstatných mien: ujo, dedo, chlap, hrdina, rozhodca, sudca, redaktor, publicista, jeleň  

D)     Neživotné podstatné mená - sú to podstatné mená mužského rodu , ktoré majú neživý charakter.

Pozor: Pri zvieracích podstatných menách mužského rodu sú neživotné len množnom čísle a zvieracie podstatné mená mužského rodu  v množnom čísle sa skloňujú podľa vzorov dub a stroj.

Príklady neživotných podstatných mien: dub, stroj, kremeň, kameň, pokoj, referát, papier, certifikát, diplom, mantinel, cukrík, jelene    

Zvieracie podstatné mená vlk, pes, býk a vták sa skloňujú v jednotnom čísle podľa vzorov chlap a hrdina a v množnom čísle sa môžu skloňovať podľa  vzorov chlap –dub.

E)      Pomnožné podstatné mená - sú podstatné mená, ktoré existujú len v množnom čísle

Príklady pomnožných podstatných mien: gate, prsia, Levice, nohavice, ponožky, pukance

Priradenie vzoru ku pomnožným podstatným menám

Napríklad máme pomnožné podstatné meno Košice.  

Slovo Košice majú koncovku e, tak skúmame, ktorý vzor podstatného mena má v nominatíve  množnom čísle  koncovku e.

Koncovku e majú v nominatíve v množnom čísle  vzory stroje, ulice, dlane z toho vyplýva, že z týchto  vzorov vyberáme.

Teraz zisťujeme pomocou genitívu plurálu, ktorý vzor podstatných mien je slovo Košice.

Je bez ulíc, je bez Košíc

Tak slovo  Košice sa skloňuje podľa ulica.

F)      Hromadné podstatné mená - sú podstatné mená, ktoré existujú len v jednotnom čísle

Príklady hromadných podstatných mien: občianstvo, lístie

G)     Vlastné podstatné mená - sú to podstatné mená, ktoré pomenúvajú názvy miest, osôb, inštitúcii, udalosti a názvy štátnych príslušníkov. Ich charakteristickým znakom je, že  začiatočne písmeno je veľké 

Príklady vlastných podstatným mien: Levice, Michal, Slovenské národné múzeum, Slovenské národné povstanie, Slovák  

H)     Všeobecné podstatné mená - sú všetky ostatné podstatné mená, začiatočne písmeno nie je veľké

Príklady všeobecných podstatných mien: dub , kabelka, strýko, tenis, hviezda

Čo určujeme pri podstatných menách:

A)      Rod - rod môže byť mužský (ten), ženský (tá) a stredný (to)

B)      Číslo - číslo môže byť jednotné ( singulár) a množné (plurál)

C)      Pád - pád môže byť nominatív - N (Kto?, Čo?), genitív - G (Koho?, Čoho?), Datív - D (Komu?, Čomu?), Akuzatív - A (Koho? Čo?), Lokál - L (Kom?, Čom?) a inštrumentál - I (Kým?, Čím?).

D)      Vzor

o    Mužský rod: chlap, hrdina, dub, stroj a kuli

o    Ženský rod: žena, ulica, dlaň, kosť a gazdiná

o    Stredný rod: mesto, srdce, vysvedčenie a dievča

E)      Životnosť - životnosť sa určuje pri podstatných menách mužského rodu a podstatné mená môžu byť životné (pri vzoroch chlap, hrdina) a môžu byť neživotné (pri vzoroch dub, stroj).

 

Charakteristické črty  vzorov skloňovania:

A)     Podstatné mená mužského rodu:

1.       Skloňovanie podľa vzoru chlap: podľa vzoru chlap sa skloňujú všetky životné podstatné mená, ktoré sú v nominatíve plurálu zakončené na samohlásku  i.

2.       Skloňovanie podľa vzoru hrdina: podľa vzoru hrdina sa skloňujú všetky životné podstatné mená, ktoré sú v nominatíve singuláru zakončené na samohlásku a.

3.       Skloňovanie podľa vzoru dub: podľa vzoru dub sa skloňujú všetky neživotné podstatné mená, ktoré sú v nominatíve plurálu zakončené na samohlásku y a v nominatíve singuláru zakončené na tvrdú spoluhlásku.

4.       Skloňovanie podľa vzoru stroj: podľa vzoru stroj sa skloňujú všetky neživotné podstatné mená, ktoré sú v nominatíve plurálu zakončené na samohlásku e a v nominatíve singuláru zakončené na mäkkú spoluhlásku.

5.       Skloňovanie podľa vzoru kuli: podľa vzoru kuli sa skloňujú všetky podstatné mená mužského rodu zakončené na samohlásku i.

B)      Podstatné mená ženského rodu:

1.       Skloňovanie podľa vzoru žena: podľa vzoru žena sa skloňujú všetky podstatné mená ženského rodu zakončené v nominatíve plurálu na y a súčasne pred samohláskou a je tvrdá spoluhláska. 

2.       Skloňovanie podľa vzoru ulica: podľa vzoru ulica sa skloňujú všetky podstatné mená ženského rodu zakončené v nominatíve plurálu  na samohlásku e a súčasne pred samohláskou a je mäkká spoluhláska.

3.       Skloňovanie podľa vzoru dlaň: podľa vzoru dlaň sa skloňujú všetky podstatné mená ženského rodu zakončené v nominatíve plurálu na e.

4.       Skloňovanie podľa vzoru kosť: podľa vzoru kosť sa skloňujú všetky podstatné mená ženského rodu zakončené v nominatíve plurálu na i.

5.       Skloňovanie podľa vzoru gazdiná: podľa vzoru gazdiná sa skloňujú všetky podstatné mená ženského rodu zakončené v nominatíve singuláru na á.

C)      Podstatné mená stredného rodu:

1.       Skloňovanie podľa vzoru mesto: podľa vzoru mesto sa skloňujú všetky podstatné mená stredného rodu zakončené v nominatíve plurálu na á.

2.       Skloňovanie podľa vzoru dievča: podľa vzoru dievča sa skloňujú všetky podstatné mená stredného rodu zakončené v nominatíve plurálu na tá.

3.       Skloňovanie podľa vzoru srdce: podľa vzoru srdce sa skloňujú všetky podstatné mená stredného rodu zakončené v nominatíve singuláru na e.

4.       Skloňovanie podľa vzoru vysvedčenie: podľa vzoru vysvedčenie sa skloňujú všetky podstatné mená stredného rodu zakončené v nominatíve singuláru na ie.    

 

Skloňovanie vzorov podstatných mien

Mužský rod

Vzor chlap

Singulár - N: chlap; G: bez  chlapa; D: dám chlapovi; A: vidím chlapa; L: o chlapovi; I: s chlapom

Plurál - N: dvaja chlapi; G: bez chlapov; D: dám chlapom; A: vidím chlapov; L: o chlapoch; I: s chlapmi

Vzor hrdina

Singulár - N: hrdina; G: bez hrdinu; D: dám hrdinovi; A: vidím hrdinu; L: o hrdinovi; I: s hrdinom

Plurál - N: dvaja hrdinovia; G: bez hrdinov; D: dám hrdinom; A: vidím hrdinov; L: o hrdinoch; I: s hrdinami

Vzor dub

Singulár - N: dub; G: bez duba; D: dám dubu; A: vidím dub; L: o dube; I: s dubom

Plurál -  N: dva duby; G: bez dubov; D:dám dubom; A: vidím duby; L: o duboch; I: s dubmi

Vzor stroj

Singulár - N: stroj; G: bez stroja; D: dám stroju; A: vidím stroj; L: o stroji; I: so strojom

Plurál - N: 2 stroje; G: bez strojov; D: dám strojom; A:vidím stroje; L:o strojoch; I: so strojmi

Vzor kuli

Singulár - N: kuli; G: bez kuliho;  D: dám kulimu; A: vidím kuliho; L: o kulim;  I: s kulim

Plurál - N: kuliovia; G: bez kuliov; D: kuliom; A: vidím kuliov; L: o kulioch; I: s kuliami

 

Skloňovanie zvieracieho podstatného mena pes:

Poznámka: Zvieracie podstatné meno pes sa skloňuje v jednotnom čísle podľa vzoru chlap:

Singulár - N: pes (chlap), G: bez psa (bez chlapa) , D: dám psovi (dám chlapovi) , A: vidím psa (vidím chlapa), L: o psovi (o chlapovi) , I: so psom (chlapom)

Poznámka:  Zvieracie podstatné meno pes sa môže skloňovať v množnom čísle aj podľa vzoru chlap aj podľa vzoru dub

Plurál: N: tie psy, tí psi (duby, chlapi), G: bez psov (bez dubov,  bez chlapov), D: dám psom (dám dubom, dám chlapom), A: vidím tie psy, vidím tých psov (vidím tie duby, vidím tých chlapov), L: o psoch (o duboch, o chlapoch), I: so psami  (s dubom, s chlapom)

Skloňovanie zvieracieho podstatného mena jeleň

Poznámka: Zvieracie podstatné meno jeleň sa skloňuje v jednotnom čísle   podľa vzoru chlap

Singulár - N: jeleň (chlap), G: bez jeleňa (bez chlapa), D: dám jeleňovi (dám chlapovi), A: vidím jeleňa (vidím chlapa), L: o jeleňovi (o chlapovi), I: s jeleňom (s chlapom)

Poznámka:  Zvieracie podstatné meno jeleň sa skloňuje v množnom čísle podľa vzoru stroj.

Plurál -  N: jelene (stroje), G: bez jeleňov (bez strojov), D: dám jeleňom (dám strojom), A: vidím jelene (vidím stroje), L: o jeleňoch (o strojoch), I: s jeleňmi (so strojmi)  

 

Ženský rod

Vzor žena

Singulár- N:žena- G:bez ženy- D:dám žene- A:vidím ženu- L:o žene- I:so ženou

Plurál- N:dve ženy- G:bez žien- D:dám ženám- A:vidím ženy- L:o ženách- I:so ženami

Vzor ulica

Singulár - N: ulica, G: bez ulice, D: dám ulici, A: vidím ulicu, L: o ulici, I: s ulicou

Plurál - N: dve ulice, G: bez ulíc, D: dám uliciam, A: vidím ulice, L: o uliciach, I: s ulicami

Vzor dlaň

Singulár - N: dlaň, G: bez dlane, D: dám dlani, A: vidím dlaň, L:o dlani, I: s dlaňou

Plurál - N: dve dlane, G: bez dlaní, D:dám dlaniam, A: vidím dlane, L: o dlaniach, I: s dlaňami    

Vzor kosť

Singulár - N: kosť, G: bez kosti, D: dám kosti, A: vidím kosť, L: o kosti, I: s kosťou

Plurál - N: dve kosti, G: bez kostí, D: dám kostiam, A: vidím kosti, L: o kostiach, I: s kosťami

Vzor gazdiná

Singulár -  N: gazdiná, G: bez gazdinej, D: dám gazdinej, A: vidím gazdinú, I: s gazdinou

Plurál - N: dve gazdiné,G: bez gazdín, D: dám gazdinám, A: vidím gazdiné, I: s gazdinami

Skloňovanie podstatných mien ženského rodu, ktoré majú koncovku ea  

Poznámka:  Podstatné mená ženského rodu Kórea, orchidea, Andrea, idea sa skloňuje v singulári prevažne podľa vzoru žena, výnimkou je datív a lokál singuláru, ktorý sa skloňuje podľa vzoru ulica.

Singulár - N: Kórea, orchidea, Andrea, idea (žena), G: bez Kórey, bez orchidey, bez Andrey, bez idey (bez ženy), D: dám Kórei, dám orchidei, dám idei, dám Andrei (dám ulici), A: vidím Kóreu, vidím orchideu, vidím ideu, vidím Andreu (vidím ženu), L: o Kórei, o orchidei, o idei, o Andrei (o ulici), I: s Andreou, s ideou, s orchideou, s Kóreou (so ženou)

Poznámka:  Podstatné mená ženského rodu Kórea, orchidea, Andrea, idea sa skloňuje v pluráli  prevažne podľa vzoru žena, výnimkou je genitív plurálu   ktorý sa skloňuje podľa vzoru dlaň.

Plurál - N: dve idey, dve Andrey, dve orchidey, dve Kórey (dve ženy), G: bez ideí, bez Andreí, bez orchideí, bez Kórei (bez dlaní), D: dám ideám, dám Andreám, dám orchideám, dám Koreám (dám ženám), A: vidím ideu, vidím Andreu, vidím orchideu, vidím Kóreu (vidím ženu), L:  o ideách, o Andreách, o orchideách, o Kóreach (o ženách), I: s ideami, s Andreami, s orchideami, s Kóreami (so ženami)    

Stredný rod

Vzor mesto

Singulár - N: mesto,  G: bez mesta,  D: dám mestu,  A: vidím mesto, L: o meste, I: s mestom

Plurál - N: dve mestá- G: bez miest, D: dám mestám, A: vidím mestá, L: o mestách, I: s mestami

Vzor srdce

Singulár - N: srdce, G: bez srdca, D: dám srdcu, A: vidím srdce, L: o srdci, I: so srdcom

Syntax - zaoberá sa vzťahmi medzi slovami vo vete, správnym tvorením vetných konštrukcií a slovosledom. Základnou jednotkou skladby je veta, ktorá sa skladá z vetných členov.

Vetný člen - vetný člen je „stavebná jednotka“ vety. Môžu to byť podstatné mená, slovesá, prídavné mená, číslovky, príslovky, zámená. Vetnými členmi nemôžu byť predložky, spojky a častice.

Syntagma -  sklad - spojenie dvoch plnovýznamových slov, z ktorých zväčša jedno je riadiace (nadradené, hlavné), druhé je riadené (podriadené, závislé).

Existujú tri základné druhy syntagiem:

Priraďovací sklad – viacnásobný vetný člen
Určovací sklad
Prisudzovací sklad (podmet - prísudok)

Polovetná konštrukcia - medzi jednoduchou vetou a súvetím sú polovetné konštrukcie. Hlavným cieľom polovetných konštrukcií je stiesňovať, skracovať text. Polovetná konštrukcia sa vyjadruje:

1. prechodníkom (opakujúc, kľačiac):
- Chodil po triede a potichu si opakoval učivo
- Chodil po triede, potichu si opakujúc učivo. 
2. činným príčastím prítomným (zaujímajúci, píšuci):
- Janko sa zaujímal o hudbu a po skončení školy si založil skupinu.
- Janko, zaujímajúci sa o hudbu, po skončení školy si založil skupinu.
3. neurčitkom (robiť, konať)

 
13:45 Polícia Alkohol
 
Päť vodičov v Nitrianskom kraji skončilo počas veľkonočných sviatkov v policajnej cele. Dôvodom bola jazda pod vplyvom alkoholu, informovalo Krajské riaditeľstvo Policajného zboru v Nitre.
13:40 Polícia Alkohol
 
Polícia obvinila vodičku, ktorá jazdila v Humennom pod vplyvom alkoholu. Výsledok jej dychovej skúšky bol 1,5 promile. Ako TASR informovala hovorkyňa Krajského riaditeľstva Policajného zboru v Prešove Jana Ligdayová, policajná hliadka ju zastavila z dôvodu nekoordinovanej jazdy.
13:39 Cyprus Izrael Austrália
 
Britský minister zahraničných vecí David Cameron vyzval v utorok Izrael na "úplné, transparentné vysvetlenie" leteckého útoku v Pásme Gazy, pri ktorom v pondelok zahynulo sedem humanitárnych pracovníkov organizácie World Central Kitchen (WCK). TASR prevzala informácie z agentúry DPA a portálu britského denníka The Guardian.
13:39
 
Chorobnosť na akútne respiračné ochorenia (ARO) v 13. kalendárnom týždni tohto roka v porovnaní s predchádzajúcim týždňom klesla o 17,2 percenta. Chorobnosť na chrípku a chrípke podobné ochorenia (CHPO) klesla o 26,4 percenta. TASR o tom informovali z odboru komunikácie Úradu verejného zdravotníctva (ÚVZ) SR.

13:38 Polícia Nehoda
 
Polícia upozorňuje motoristov, že cesta medzi obcami Popudinské Močidľany a Radošovce v okrese Skalica je uzavretá. Dôvodom je vážna dopravná nehoda, informuje Krajské riaditeľstvo Policajného zboru v Trnave.
13:37
 
Koncom tohto roka by mohlo byť v Ozbrojených silách (OS) SR 20.982 vojakov a 4500 zamestnancov. Ide o zvýšenie o 279 vojakov oproti aktuálne schválenému počtu. Vyplýva to z návrhu aktualizácie početných stavov, ktorý Ministerstvo obrany (MO) SR predložilo do medzirezortného pripomienkového konania.
13:35
 
Gemersko-malohontské múzeum (GMM) v Rimavskej Sobote pripravilo pri príležitosti finisáže výstavy Dress kód: Art nouveau pre verejnosť dvojicu odborných prednášok. Návštevníkom detailne priblížia ženskú módu v období secesie a Belle Époque. TASR o tom informovala PR manažérka múzea Szilvia Tóth.
13:34
 
V Krompachoch kompletne zrekonštruujú Základnú školu (ZŠ) s materskou školou (MŠ) na Maurerovej ulici, mestu sa podarilo získať dotáciu z Plánu obnovy a odolnosti SR. Primátor Dárius Dubiňák pre TASR uviedol, že vďaka projektu v hodnote 2,76 milióna eur sa škola zmení na modernú inštitúciu hodnú 21. storočia.
13:34 NATO
 
Šéf slovenskej diplomacie Juraj Blanár sa v stredu a vo štvrtok zúčastní na rokovaní ministrov zahraničných vecí členských štátov NATO v Bruseli. TASR o tom informoval komunikačný odbor Ministerstva zahraničných vecí a európskych záležitostí (MZVEZ) SR. Na stretnutí sa prvýkrát zúčastní v pozícii plnohodnotného spojenca aj Švédsko, ktoré sa 7. marca oficiálne stalo 32. členským štátom NATO.
13:32 Švajčiarsko
 
Švajčiarska banka UBS oznámila, že spúšťa ďalšie kolo spätného odkúpenia akcií, a to v hodnote 2 miliardy USD (1,85 miliardy eur). Zároveň dodala, že akcie za približne polovicu uvedenej hodnoty plánuje spätne odkúpiť do konca roka. Informovala o tom agentúra Reuters.
13:30 EÚ Izrael
 
Predsedníčka Európskej komisie Ursula von der Leyenová v utorok vyjadrila sústrasť rodinám a priateľom humanitárnych pracovníkov organizácie World Central Kitchen, ktorí prišli o život v pondelok počas leteckého útoku Izraela v Pásme Gazy, informuje TASR.
13:27 NATO
 
Šéf slovenskej diplomacie Juraj Blanár (Smer-SD) sa v najbližšie dni (3. - 4. 4.) zúčastní na rokovaní ministrov zahraničných vecí členských štátov NATO v Bruseli. TASR o tom informoval komunikačný odbor Ministerstva zahraničných vecí a európskych záležitostí (MZVEZ) SR. Na stretnutí sa prvýkrát zúčastní v pozícii plnohodnotného spojenca aj Švédsko, ktoré sa 7. marca oficiálne stalo 32. členským štátom NATO.

13:25 Holandsko Futbal
 
Holandský futbalový klub Ajax Amsterdam v utorok oznámil, že pozastavuje výkonnú funkciu generálnemu riaditeľovi a predsedovi predstavenstva Alexovi Kroesovi. Ten čelí podozreniam z obchodovania s dôvernými informáciami. Informovala o tom agentúra AFP.
13:22 Kultúra
 
Začínajúci profesionálni maliari sa môžu prihlásiť do 19. ročníka súťaže Maľba. Prihlášky môžu posielať do 25. mája. TASR o tom informovali z Nadácie VÚB, ktorá súťaž organizuje.
13:21
 
Bratislavská mliekareň Rajo mení názov na Meggle Slovakia, výrobky pod značkou Rajo však pokračujú. Informovala o tom spoločnosť.
13:09 Polícia
 
Polícia obvinila 14-ročného chlapca, nožom mal bodnúť do krku muža v obci Svinia. Krajská policajná hovorkyňa z Prešova Jana Ligdayová informovala, že incident sa stal v sobotu (30. 3.) popoludnia.

13:07 Rusko Ukrajina
 
Ukrajinské drony zasiahli pri utorkovom útoku v Tatársku v Rusku jednu z najväčších ruských ropných rafinérii. TASR o tom informuje podľa správy agentúry Reuters.
13:04
 
Nápor na daňové úrady vrcholí, v utorok je posledný deň, keď si subjekty musia splniť svoje povinnosti voči finančnej správe (FS). Daňovníci tento rok môžu podať daňové priznanie do 2. apríla. Uviedol to Radoslav Kozák, vedúci komunikačného oddelenia Finančného riaditeľstva SR.
12:58 Polícia Nehoda
 
V Starej Ľubovni zrazila 62-ročná vodička staršiu chodkyňu hneď dvakrát, prípadom sa zaoberá polícia. Krajská policajná hovorkyňa z Prešova Jana Ligdayová informovala, že nehoda sa stala ešte minulý týždeň v stredu (27. 3.) podvečer na Okružnej ulici.
12:54 Polícia
 
Polícia obvinila agresívneho muža, ktorý mal uniesť a fyzicky napadnúť ženu v obci Spišský Hrhov neďaleko Levoče. Krajská policajná hovorkyňa z Prešova Jana Ligdayová informovala, že incident sa stal v nedeľu (31. 3.) v noci.
12:48
 
Egyptský prezident Abdal Fattáh Sísí v utorok zložil pred parlamentom prísahu na svoje tretie funkčné obdobie. Pri moci je už desať rokov a prezidentom má byť až do roku 2030, informuje TASR s odvolaním sa na agentúru AFP.
12:46 Polícia Pátranie
 
Polícia pátra po zlodejovi, ktorý z rodinného domu v Levoči ukradol tisícky eur. Krajská policajná hovorkyňa z Prešova Jana Ligdayová informovala, že neznámy páchateľ sa do domu na Štúrovej ulici vlámal počas uplynulého víkendu (30. a 31. 3.).
12:39 Futbal
 
Český futbalový klub FC Slovan Liberec má od apríla nového majiteľa. Tri štvrtiny klubových akcií prevzal mladý podnikateľ Ondřej Kania. Generálnym riaditeľom spoločnosti vlastniacej 75,65 percenta akcií sa stal bývalý útočník Jan Nezmar.
12:35 USA Turecko
 
Americký výrobca domácich spotrebičov Whirlpool oznámil, že dokončil transakciu s tureckou spoločnosťou Arcelik, výsledkom ktorej je nový podnik na výrobu spotrebičov v Európe. V ňom bude väčšinovým vlastníkom turecká firma. TASR o tom informuje na základe správy agentúry DPA.
12:32 Nemecko
 
Nemecká sieť obchodných domov Galeria Karstadt Kaufhof (GKK), ktorá začiatkom januára podala na okresnom súde v Essene návrh na vyhlásenie platobnej neschopnosti, rokuje s novými potenciálnymi investormi. Oznámil to v utorok súd. TASR o tom informuje na základe správy DPA.

Uveďte typické znaky všetkých členov a syntagiem prostredníctvom ukážky:
Biele dvere izby sa prudko otvorili. 
Chodil po miestnosti, spievajúc si. 
Chodil po miestnosti a spieval si.
Zašli sme do domu vypýtať si vodu.

Zašli sme do domu, aby sme si vypýtali vodu.

Biele dvere izby sa prudko otvorili: 
biele - zhodný prívlastok
dvere – podmet
izby – nezhodný prívlastok
sa otvorili – prísudok
prudko – príslovkové určenie spôsobu

Plurál -  N: dve srdcia, G: bez sŕdc, D: dám srdciam, A:vidím srdciam, L:o srdciach, I: so srdcami
Frazeologizmy sú viacslovné ustálené spojenia, ktoré pomenúvajú skutočnosť nepriamo (obrazne) – nechápeme ich doslovne. Do tejto skupiny patria:

príslovia – prinášajú nejaké ponaučenie, napr. Bez práce nie sú koláče. alebo Kto vysoko lieta, nízko padá. 
porekadlá – narozdiel od príslov neprinášajú žiadne ponaučenie, len konštatujú, napr. Zíde z očí, zíde z mysle. alebo Má zlato v hrdle. 
prirovnania – prirovnáva jednu vec alebo osobu ku druhej veci alebo osobe, napr. Leží ako zabitý. alebo Vlečie sa sťa slimák.
pranostiky – podľa dlhodobého pozorovania počasia a úrody našich predkov boli sformované pranostiky, ktoré hovoria o počasí alebo úrode, napr. Katarína na blate, Vianoce na ľade. alebo Studený máj, v stodole raj.

Vzor vysvedčenie

Singulár - N: vysvedčenie, G: bez vysvedčenia, D: dám vysvedčeniu, A: vidím vysvedčenie, L: o vysvedčení, I: s vysvedčením

Plurál - N: dve vysvedčenia, G: bez vysvedčení, D: dám vysvedčeniam, A: vidím vysvedčenia, L: o vysvedčeniach, I: s vysvedčeniami

Vzor dievča

Singulár - N: dievča, G: bez dievčaťa, D: dám dievčaťu, A: vidím dievča, L: o dievčati, I: s dievčaťom

Plurál - N: dve dievčatá, G: bez dievčat, D: dám dievčatám, A: vidím dievčatá, L: o dievčatách, I: s dievčatami  

Singulár - N: páví, pávia, pávie, G: bez pávieho, bez pávej, bez pávieho, D: dám páviemu, dám pávej, dám páviemu, A: vidím pávieho, vidím páviu, vidím pávie, L: pávom, pávej, pávom, I: s pávím, s pávou, s pávím

Plurál - N: páví, pávie, pávie, G: bez pávích, D: dám pávím, A: vidím pávích, vidím pávie, vidím pávie, L: o pávích, I: s pávími

Vzor matkin

Singulár - N: matkin, matkina, matkine, G: bez matkinho, bez matkinej, bez matkinho, D: dám matkinmu, dám matkinej, dám matkinmu, A: vidím matkinho, vidím matkinu, vidím matkine, L: o matkinom, o matkinej, o matkinom, I: s matkiným, s matkinou, s matkiným

Plurál - N: dvaja matkini, dve matkine, dve matkine, G:  bez matkiných, D: dám matkiným, A: vidím matkiných,  vidím matkine, vidím matkine, L: o matkiných, I: s matkinými

Vzor otcov

Singulár - N: otcov, otcova, otcovo, G: bez otcovho, bez otcovej,  bez otcovho, D: dám otcovmu, dám otcovej, dám otcovmu, A: otcovho, otcovu, otcovo, L: o otcovom, o otcovej, o otcovom, I: s otcovým, s otcovou, s otcovým

Plurál - N: dvaja otcovi, dve otcove, dve otcove, G: bez otcových, D: dám otcovým, A: vidím otcových, vidím otcove, vidím otcove, L: o otcových, I: s otcovými

Lomká má v systéme Unicode kód U+002F SOLIDUS. $@$@$

Vtáčik veselo letel navôkol a už ho nič nebolelo! A nie len to! Vtáčik Bobík  sa už veľkého medveďa viac nebál a z dvoch síce rozdielnych zvieratiek sa stali nerozluční priatelia. Vtáčik celú zimu nosil spiacemu medvedíkovi jedlo, ktoré nosil ako vďaku za to, že mu pomohol. Pretože nie je dôležité ako vyzeráš, ale aký si vnútri. `


let text = `Skúsme Tokenizovať takúto vetu možno toto dopadne lepšie, alebo ja neviem skúsim alexandrinú správu tokenizovať Zase ma strasne moc vnímaš Ondrej To naco robíš?? Pičovina táto slovenčina`
const corpus = corpusText.match(regex)
console.log(corpus)


const bpeTokenizer = new BPETokenizer();
bpeTokenizer.learnVocab(corpus, 50);
bpeTokenizer.purgeVocabulary(3000); 


console.log("BPE Vocabulary Size:", bpeTokenizer.vocab.length);
console.log("Tokenized text:", JSON.stringify(bpeTokenizer.tokenize(text)));


const encodedText = bpeTokenizer.encode(text);
console.log("Encoded:", encodedText);

const decodedText = bpeTokenizer.decode(encodedText);
console.log("Decoded:", decodedText);
