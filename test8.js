
class Vocab {
    constructor() {
        this.vocab = [];
    }

    build(inputString, vocabSize = 1000) {
        // Step 1: Initialize vocabulary with individual characters
        const vocab = new Set(inputString.split(''));

        // Step 2: Perform Byte Pair Encoding
        while (vocab.size < vocabSize) {
            // Step 2a: Count pairs frequency
            const pairs = {};
            inputString.split('').forEach((letter, index) => {
                if (index < inputString.length - 1) {
                    const pair = inputString.slice(index, index + 2);
                    if (!pairs[pair]) {
                        pairs[pair] = 0;
                    }
                    pairs[pair]++;
                }
            });

            // Step 2b: Find the most frequent pair
            let mostFrequentPair = null;
            let maxFrequency = 0;
            for (const pair in pairs) {
                if (pairs[pair] > maxFrequency) {
                    mostFrequentPair = pair;
                    maxFrequency = pairs[pair];
                }
            }

            // Step 2c: Merge the most frequent pair
            if (mostFrequentPair) {
                console.log(mostFrequentPair);
                vocab.add(mostFrequentPair);
                inputString = inputString.split(mostFrequentPair).join('');
            } else {
                // No more pairs to merge, break the loop
                break;
            }
        }

        // Step 3: Update vocabulary
        this.vocab = Array.from(vocab);
    }

    tokenize(str) {
        return str.split('').map((letter) => this.vocab.indexOf(letter) / (this.vocab.length - 1));
    }
     tokenizeChar(char) {
        return this.vocab.indexOf(char) / (this.vocab.length - 1);
    }

    detokenize(tokens) {
        return tokens.map((token) => this.vocab[Math.round(token * (this.vocab.length - 1))]).join('');
    }
}


// Define a trigram Bayesian network class
const fs = require('fs'); // Include the file system module
const axios = require('axios'); // Include the file system module

// Define a trigram Bayesian network class





// Example usage
const corpus = `🗑️Hlavní menu
 
WikipedieWikipedie: Otevřená encyklopedie

Hledat
Vytvoření účtu
Přihlášení

Osobní nástroje
Obsah  skrýt
(úvod)
Osobní život
Přepnout podsekci Osobní život
Komunistická strana Slovenska
Strana demokratické levice
SMER – sociálna demokracia
Integrace levicových stran
Volební vítězství
Parlamentní volby 2006 a první vláda
Parlamentní volby 2010 a opozice
Parlamentní volby 2012 a druhá vláda
Prezidentská kandidatura v roce 2014
Parlamentní volby 2016 a třetí vláda
V opozici
Parlamentní volby 2023 a čtvrtá Ficova vláda
Názory na Ukrajinu
Kontroverze
Přepnout podsekci Kontroverze
Kauza Očistec a Súmrak
Onemocnění covid-19
Ocenění
Odkazy
Přepnout podsekci Odkazy
Reference
Související články
Externí odkazy
Robert Fico

58 jazyků
Článek
Diskuse
Číst
Zobrazit zdroj
Zobrazit historii

Nástroje
 Tato stránka je zamčena pro neregistrované a nové uživatele
Možná hledáte: Roberto Fico – italský politik.
doc. JUDr. Robert Fico, CSc.
Robert Fico (2023)
Robert Fico (2023)
5., 7. a 12. premiér Slovenska
Úřadující
Ve funkci od:
25. října 2023
Prezidentka	Zuzana Čaputová
Předchůdce	Ľudovít Ódor
Ve funkci:
4. července 2006 – 8. července 2010
Prezident	Ivan Gašparovič
Předchůdce	Mikuláš Dzurinda
Nástupkyně	Iveta Radičová
Ve funkci:
4. dubna 2012 – 22. března 2018
Prezident	Ivan Gašparovič
Andrej Kiska
Předchůdkyně	Iveta Radičová
Nástupce	Peter Pellegrini
1. předseda SMER–SD
Úřadující
Ve funkci od:
8. listopadu 1999
Předchůdce	subjekt vznikl
Poslanec Národní rady SR
Úřadující
Ve funkci od:
1. října 1992
(od 4. 7. 2006 do 8. 7. 2010 a od 4. 4. 2012 do 22. 3. 2018 se mandát neuplatňoval)
Stranická příslušnost
Členství	KSS (1987–1990)
SDĽ (1990-1999)
SMER–SD (od 1999)
Narození	15. září 1964 (59 let)
Topoľčany
Československo Československo
Národnost	slovenská
Choť	Svetlana Ficová
Děti	Michal
Alma mater	Právnická fakulta Univerzity Komenského v Bratislavě
Profese	právník
Náboženství	katolicismus
Ocenění	Stužka řádu Bílého lva první třídy Řád Bílého lva I. tř.
Podpis	Robert Fico, podpis
Commons	Robert Fico
Některá data mohou pocházet z datové položky.
Robert Fico (* 15. září 1964 Topoľčany) je slovenský politik, od října 2023 předseda vlády Slovenské republiky, když tuto funkci zastával již v letech 2006–2010 a 2012–2018. Je zakladatelem strany SMER – sociálna demokracia a jejím prvním předsedou.
Působil i jako vysokoškolský učitel na Právnické fakultě Univerzity Komenského v Bratislavě. V roce 2002 se habilitoval v oboru bezpečnostní služby a získal docenturu na Akademii Policejního sboru v Bratislavě.[1] V roce 2013 oznámil svou kandidaturu na prezidenta Slovenské republiky. Ve volbách na jaře 2014 však byl ve druhém kole poražen Andrejem Kiskou.
Osobní život
Narodil se v Topoľčanech jako druhý syn v rodině Ľudovíta Fica, řidiče vysokozdvižného vozíku, a matky Emílie Ficové, která pracovala jako prodavačka obuvi. Má staršího bratra Ing. Ladislava Fica, podnikatele ve stavebnictví, a o čtrnáct let mladší sestru Lucii Chabadovou, která v současnosti působí jako právnička. Do věku šesti let žil i s rodinou v obci Hrušovany, poté se přestěhovali do Topoľčan.
Se svou manželkou Svetlanou, která je právnička a vysokoškolská pedagožka (profesorka), se setkal během studií práv v Bratislavě. Mají spolu syna jménem Michal.
V roce 1986 absolvoval Právnickou fakultu Univerzity Komenského v Bratislavě a získal titul JUDr. V roce 1992 získal titul CSc. s prací na téma „Trest smrti v Československu“. Na počátku 90. let studoval v rámci Masarykova stipendia na Škole slovanských a východoevropských studií UCL v Londýně.[2] V roce 2002 ukončil postgraduální studium a získal titul docent.
Od absolvování školy až do roku 1995 pracoval na Právnickém institutu Ministerstva spravedlnosti SR. V letech 1994-2000 působil jako agent pro zastupování SR v řízení před Evropskou komisí pro lidská práva a Evropským soudem pro lidská práva.
V lednu 2024 si koupil luxusní byt v Bratislavě od svého stranického kolegy Dušana Muňka, který si od něj tři roky pronajímal. Kupní cena bylo o třetinu nižší než cena tržní.[3][4][5] V letech 2012 až 2019 si pronajímal luxusní byt od kontroverzního podnikatele Ladislava Bašternáka.[3][6] Z toho se odstěhoval až po tom, kdy by Bašternák pravomocně odsouzen za daňové podvody a součástí jeho trestu bylo i propadnutí majetku.[7][6]
Komunistická strana Slovenska
14. dubna 1987 vstoupil do Komunistické strany Slovenska (KSS), do které byl vybrán na základě výborných studijních výsledků, ambicí i vhodného původu.
Strana demokratické levice
V roce 1992 byl ve volbách zvolen do funkce poslance za Stranu demokratické levice (SDĽ), která vznikla po listopadu 1989 přejmenováním tehdejší KSS. Od té doby byl až do roku 2006 poslancem Národní rady Slovenské republiky. Po volbách v roce 1994, ve kterých SDĽ ztratila velkou část podle průzkumů očekávaných hlasů, odstoupil z postu předsedy strany Peter Weiss a za svého nástupce následně označil Roberta Fica. Několik hodin před začátkem sjezdu však Fico kandidaturu stáhl ve prospěch Ľubomíra Fogaše, svého bývalého kolegy z Právnické fakulty Univerzity Komenského.
Po volbách v roce 1998 vznikala široká koalice stran SDK, SDĽ, SMK a SOP. Robert Fico, jakožto člen vládní SDĽ začal proti této koalici vystupovat, především útočil na spolupráci se Stranou maďarské koalice (SMK), když tvrdil, že se strana pokouší otvírat tzv. Benešovy dekrety.
V té době byl ve své straně nejpopulárnějším politikem a ve volbách získal rovněž nejvíce preferenčních hlasů z politiků SDĽ. Kvůli neshodám stranu však následně opustil.
SMER – sociálna demokracia

Fico a Boris Tadić, prezident Srbska, 2008
V prosinci 1999 založil novou politickou stranu s názvem SMER. Hned po svém založení začal SMER působit jako alternativa jak vůči tehdejší vládní koalici pod vedením Mikuláše Dzurindy, tak vůči opozici. Ještě v témže volebním období (1998-2002) přijala strana do svého názvu přívlastek "třetí cesta". Podle tvrzení představitelů strany se tím definovala jako strana moderního progresivního středolevého politického proudu typu britských labouristů nebo německé SPD.
V únoru 2003 se postavil proti plánované americké invazi do Iráku a kritizoval vládu Mikuláše Dzurindy, která se rozhodla do války zapojit, se slovy, že "Slovenská vláda je až příliš horlivá při prosazování cizích zájmů. Postup vlády je předčasný a neevropský."[8] Fico jako slovenský premiér prosadil stažení slovenských vojáků z Iráku.[9]
Preference strany postupně rostly a naopak preference SDĽ klesaly. Tehdy vládní strana SDĽ se v následujících volbách do parlamentu nedostala, a k 1. lednu 2005 zanikla, když byla pohlcena právě SMERem. V těchto volbách byl Fico opět zvolen poslancem NR SR, kde se stal členem Výboru NR SR pro lidská práva, národnosti a postavení žen. Strana SMER se stala se ziskem 13,6% voličských hlasů třetí nejsilnější politickou stranou na Slovensku, za ĽS-HZDS a SDKÚ-DS a obsadila tak v Národní radě 25 poslaneckých křesel. Naproti tomu, průzkumy veřejného mínění dlouhodobě předpovídaly pro stranu lepší výsledek. SMERu se po volbách do vlády dostat nepodařilo, tu nakonec vytvořil opětovně Mikuláš Dzurinda z SDKÚ-DS (druhá vláda Mikuláše Dzurindy) a SMER zůstal v opozici. Během celého volebního období soupeřil SMER s ĽS-HZDS o vedoucí postavení v opozici. V průzkumech veřejného mínění následně SMER dosud dlouhodobě nejpopulárnější stranu ĽS-HZDS předstihl.
Integrace levicových stran
V roce 2004 se Ficovi podařilo uskutečnit projekt sjednocení levicových stran. Tři levicové strany s dlouhodobě zanedbatelnými volebními preferencemi Strana demokratické levice (SDĽ), Sociálně demokratická alternativa (SDA) a Sociálně demokratická strana Slovenska (SDSS) se tak dohodly na společné integrací se SMERem.
Integraci schválily sněmy jednotlivých stran na podzim roku 2004 a od 1. ledna 2005 strany SDĽ, SDA a SDSS zanikly. SMER si následně změnil název na současný "SMER - sociálna demokracia" (česky "SMĚR - sociální demokracie"). Významný politický přínos pro SMER z této integrace byl zisk nových voličů a získání značky sociální demokracie. Strana se následně stala členem Strany evropských socialistů.
Volební vítězství

Fico při návštěvě ruského prezidenta Dmitrije Medveděva v Bratislavě, 2010

Fico, Donald Tusk, Angela Merkelová a Jean-Claude Juncker na summitu Evropské rady v Bratislavě 16. září 2016

Představitelé zemí V4 a izraelský premiér Benjamin Netanjahu na summitu v Budapešti 19. července 2017
Parlamentní volby 2006 a první vláda
Související informace naleznete také v článcích Parlamentní volby na Slovensku 2006 a První vláda Roberta Fica.
V červnu 2006 SMER-SD vyhrál parlamentní volby se ziskem 29,1 % hlasů a utvořil vládní koalici s ĽS-HZDS Vladimíra Mečiara a SNS Jána Sloty. Obě zmiňované strany tvořily v období 1994–1998 koaliční vládu. Vláda měla 16 členů, z nichž 11 získal SMER-SD, 3 SNS a 2 ĽS-HZDS. Ve vládě zastával post jejího předsedy a v krátkém období na přelomu června a července 2009 byl rovněž dočasně pověřeným ministrem spravedlnosti Slovenska.
Fico odmítl uznat jednostranné vyhlášení nezávislosti Kosova na Srbsku v únoru 2008.[10][11]
Parlamentní volby 2010 a opozice
Související informace naleznete také v článku Parlamentní volby na Slovensku 2010.
I když ve volbách v červnu 2010 zvítězila strana SMER-SD se ziskem 34,79 % (oproti minulým volbám si tak strana polepšila o necelých 6 % a získala více než dvojnásobný počet hlasů než druhá v pořadí SDKÚ-DS), nebyl Fico schopen sestavit koaliční vládu. Prezident Ivan Gašparovič následně sestavením vlády pověřil Ivetu Radičovou z SDKÚ-DS, jejíž strana skončila ve volbách druhá se ziskem 15,42 %.[12]
Opakovala se tak situace, která již na Slovensku nastala po volbách 1998 a 2002, kdy zvítězila HZDS, avšak Vladimír Mečiar nebyl schopen vládu sestavit. Pověřen tak byl vždy předseda druhé nejsilnější strany (v obou případech Mikuláš Dzurinda).
Parlamentní volby 2012 a druhá vláda
Související informace naleznete také v článcích Parlamentní volby na Slovensku 2012 a Druhá vláda Roberta Fica.
Po předčasných parlamentních volbách, ve kterých strana SMER-SD zvítězila ještě s výraznějším odstupem od ostatních stran než ve volbách minulých, sestavila vládu opět v čele s Robertem Ficem. Robertu Ficovi se tak podařilo vytvořit vládu, ve které zasedli pouze členové SMER-SD a nestraníci za tuto stranu nominovaní.
Prezidentská kandidatura v roce 2014
Během projevu, ve kterém 18. prosince 2013 před Národní radou hodnotil působení své druhé vlády, oznámil kandidaturu na post prezidenta Slovenské republiky ve volbách na jaře 2014, o které se spekulovalo již několik měsíců předtím. Jeho kandidaturu už předtím schválilo vedení strany SMER-SD.[13]
V prvním kole voleb, konaném 15. března 2014, získal 28 % hlasů a vyhrál, postoupil do druhého kola.[14]
Ve druhém kole, konaném 29. března 2014, však byl poražen kandidátem Andrejem Kiskou, když získal pouze 40 %, kdežto Kiska získal více než 59 % hlasů.[15]
Parlamentní volby 2016 a třetí vláda
Související informace naleznete také v článcích Parlamentní volby na Slovensku 2016 a Třetí vláda Roberta Fica.
V roce 2016 se přihlásil k „tvrdému jádru“ EU a začal prosazovat užší spolupráci s Německem a Francií. Fico prohlásil: „Být v jádru s Německem a Francií, to je podstata mé politiky. Pro Slovensko není visegrádská čtyřka alternativou EU. V4 není pro Slovensko tím životním prostorem, který si představujeme do budoucna. Náš životní prostor je v Unii.“[16]
V listopadu 2016 zrušila Čína bilaterální jednání čínského premiéra Li Kche-čchianga a Fica před summitem zemí střední a východní Evropy a Číny v Rize. Důvodem mohlo být setkání slovenského prezidenta Andreje Kisky s dalajlamou.[17]
Ficova vláda se ocitla v krizi po vraždě slovenského novináře Jána Kuciaka, který psal o působení lidí blízkých italské mafii 'Ndrangheta na Slovensku a o jejich vazbách na slovenské politiky.[18] Fico se dostal do konfliktu se slovenským prezidentem Andrejem Kiskou, který v reakci na vraždu vyzval k obměně vlády nebo předčasným volbám.[19] Fico obvinil Kisku, že se „věnuje mocenským hrám, dělá opoziční politiku“ a „destabilizuje Slovensko“.[20] Robert Fico podal dne 15. března 2018 demisi svoji a tedy i celé vlády. Ficova vláda ukončila svoje působení po jmenování vlády Petera Pellegriniho dne 22. března 2018.
V opozici
Od roku 2018 působil v opozici. V roce 2020 se od jím vedené strany SMER-SD odloučila pro-evropská stranická frakce vedená Petrem Pellegrinim, který následně založil stranu HLAS-SD. V roce 2023 Fico zopakoval svůj dřívější postoj,[21] že anexe Krymu Ruskou federací byla provedena v rozporu s mezinárodním právem. Zároveň prohlásil, že znovuzískání Krymu Ukrajinou by rusko-ukrajinský konflikt nevyřešilo.[22] Před předčasnými parlamentními volbami na Slovensku v roce 2023 důrazně odmítal další vojenskou podporu Ukrajiny během ruské invaze na Ukrajinu, která podle něj vede k prodlužování války a ke "zbytečným a obrovským ztrátám na lidských životech",[23] a prohlásil, že je především potřeba jednat o příměří a zastavení bojů.[23][22] Podobný názor zastává i velká část obyvatel Slovenska.[24] Ostře se vyjadřoval také ke slovenské prezidentce Zuzaně Čaputové, kterou označil za "americkou agentku". Volební průzkumy z května 2023 uváděly, že SMER-SD může se ziskem přibližně 15–20% hlasů volby vyhrát a sestavit tak budoucí vládu, v níž by se Fico stal opět premiérem.[25]
Parlamentní volby 2023 a čtvrtá Ficova vláda
V parlamentních volbách konaných 30. září 2023 získala Ficem vedená kandidátka strany SMER – sociálna demokracia 22,94 % hlasů a získala 42 poslaneckých mandátů. 16. října 2023 oznámil Fico, že s předsedy dalších dvou politických stran HLAS-SD a SNS Peterem Pellegrinim a Andrejem Dankem podepsal koaliční dohodu. [26][27] Koaliční vláda na tomto půdorysu byla 25. října jmenována prezidentkou republiky Zuzanou Čaputovou. Parlamentní důvěru vláda získala 21. listopadu, když ji po čtyři dny trvající rozpravě o jejím programovém prohlášení podpořilo 78 ze 143 přítomných poslanců Národní rada Slovenské republiky. Vláda přislíbila kromě jiného zvýšení daní pro banky, bohaté firmy a lidi s vyššími příjmy, zavedení plnohodnotného 13. důchodu či zastavení státní vojenské pomoci Ukrajině bránící se ruské vojenské invazi.[28]
Koncem listopadu 2023 Fico prosadil vládní opatření, které uložilo zajistit zákaz dovozu vybraných zemědělských produktů či potravin z Ukrajiny a zavedení zvláštního režimu tranzitu zásilek s určeným zbožím přes Slovensko. Podle dokumentu z jednání vlády se opatření budou týkat pšenice, kukuřice, řepky a slunečnicových semen, medu, pšeniční mouky, sladu, třtinového a řepného cukru.[29]
Počátkem prosince Ficova vláda schválila návrh zrušit elitní složku prokuratury (Úřad speciální prokuratury) dohlížející na vyšetřování nejzávažnějších kriminálních případů, včetně korupčních kauz z doby předchozího Ficova vládnutí.[30]
V prosinci 2023 prosadil tzv. konsolidační balíček, který obsahuje např. zvýšení některých daní a zdravotních odvodů, správních a soudních poplatků, zavedení zvláštní bankovní daně a oslabení druhého důchodového pilíře. Fico ho odůvodnil vysokým deficitem veřejných financí a zadlužeností Slovenska, ale zároveň prosadil vyplacení mimořádného příspěvku k důchodům, kterým zvýšil výdaje o 438 milionů eur. Podle Fica má baliček v roce 2024 přinést do státní rozpočtu asi 1,5 miliardy eur.[31][32]
Ficova vláda také navrhla v rámci úsporných opatření na rok 2024 omezit rozpočet Rozhlasu a televize Slovenska (RTVS) o 30 procent. Šest mezinárodních novinářských organizací, včetně Reportérů bez hranic, Evropské vysílací unie a Mezinárodního tiskového institutu (IPI), vyjádřilo obavy, že plán vlády může ohrozit nezávislost RTVS a svobodu tisku v zemi. V otevřeném dopise varovaly před záměrem rozdělit RTVS na samostatnou televizi a rozhlas. Připomněly také výrok Fica, který v předvolební kampani vyhrožoval, že řediteli RTVS půjde po krku.[33]
V březnu 2024 vyzval k rezignaci předsedu Ústavního soudu Ivana Fiačana poté, co měl informovat slovenská média o verdiktu soudu dříve než strany sporu. Šlo o kauzu, kdy prezidentka Zuzana Čaputová podala v únoru 2024 k ústavnímu soudu stížnost v souvislosti s vládní novelou trestního zákoníku.[34][35] Ústavní soud poté část novely pozastavil.[36]
Názory na Ukrajinu
V říjnu 2023 prohlásil, že kořeny ruské agrese na Ukrajině je vraždění ruských civilistů ukrajinskými neonacisty.[37] V lednu 2024 uvedl, že Ukrajina není suverénním státem[38] a o měsíc později doplnil, že by jako slovenský premiér blokoval její přijetí do NATO.[39] Ve videu zveřejněném na sociálních sítích tentýž měsíc uvedl, že je ruský prezident Vladimir Putin „falešně démonizován“.[40] Na počátku roku také tvrdil, že v hlavním městě Kyjevě žádná válka neprobíhá.[41] Po jednání s ukrajinským premiérem Denysem Šmyhalem v dubnu 2024 své postoje náhle zcela změnil, když uvedl, že probíhající ruská agrese proti Ukrajině nebyla vyprovokovaná a dodal, že podporuje územní celistvost země včetně Krymského poloostrova a území Donbasu.[42][43]
Kontroverze
Kauza Očistec a Súmrak
V říjnu 2021 se ve slovenských médiích objevily odposlechy slovenské policie ze schůzek v lovecké chatě, kterých se kromě Fica účastnil i tehdejší ministr vnitra Robert Kaliňák, advokát Marek Para, syn tehdejšího policejního prezidenta Tibora Gašpara a otec kontroverzního podnikatele Norberta Bödöra. Při schůzkách měli probírat mj. aktuální trestní kauzy a trestní řízení.[44][45] V rámci kauzy, nazvané slovenskou policií jako akce Očistec, byl listopadu 2020 zadržen a obviněn Norbert Bödör, Marek Para, Tibor Gašpar a další bývalí vysoce postavení policisté.[46]
V dubnu 2022 slovenská policie v rámci související kauzy Súmrak spolu s dalšími osobami Fica obvinila ze založení zločinecké skupiny a ohrožení daňového tajemství. Spolu s ním byl obviněn také jeho stranický kolega Robert Kaliňák. Na rozdíl od Kaliňáka nebyl Fico zadržen, když k tomuto je třeba souhlas slovenského parlamentu.[47][48] Skupinu měl vést Norbert Bödör a Tibor Gašpar. Fico s Kaliňákem měli skupinu politicky krýt a využívat proti politickým oponentům, mj. proti tehdejšímu prezidentovi Andrejovi Kiskovi[49] nebo Igoru Matovičovi.[50]
V rámci vyšetřování jeden z klíčových svědků vypověděl, že na pokyn Norberta Bödöra rozesílal novinářům účetní doklady Kiskovy firmy. Robertovi Kaliňákovi na úniku dokumentů také záleželo, Fico si celou kauzu nechával podrobně vysvětlovat a úředníci Finanční správy mu účetní materiály také nosili.[50]
V listopadu 2022 slovenský generální prokurátor Maroš Žilinka celé stíhání zrušil a vyšetřování zastavil. Vyžil k tomu na Slovensku nechvalně známý paragraf 363 tamního trestního řádu.[51] Náměstek prokurátora k tomu uvedl, že konstrukce skutků byla tak neurčitá, že obviněným neumožňovala využít práva na obhajobu.[52] Stíhání bylo ve fázi těsně před podáním obžaloby.[50]
Onemocnění covid-19
Robert Fico patří k výrazným kritikům opatření proti koronaviru a odpůrcům očkování proti covid-19, za kterým je podle něj jen „prachsprostý byznys“. Z policejních nahrávek pořízených na chatě v obci Veľký Ďur v Nitranském kraji vyplývá, že Fico se v létě 2021 nemocí nakazil. Na dovolené na Krétě měl dva dny horečky, třetí den zkolaboval. Byl hospitalizován a v nemocnici se prokázalo, že má covid. Podepsal reverz a přesunul se do hotelové karantény. „Po jedenácti dnech jsem zfalšoval covidový test,“ říká na nahrávce Fico s vysvětlením, že PCR testy mu na rozdíl od antigenních stále vycházely pozitivní. „Tak jsem jel načerno z Řecka domů,“ dodal. Po návratu ze zahraničí pak nedodržel povinnou 14denní karanténu.
„Neumím se zbavit toho covidu... Celý pátek jsem ležel, v sobotu jsem zrušil celý program. Únava, stále mám plechovou chuť v ústech, jako kdybych něco necítil. Pořád srdeční arytmie a tyto hovadiny. To je strašná choroba, strašná,“ stěžoval si při dalším setkání na chatě, kde se scházel s podnikateli a advokáty, napojenými na zatčené z policejní akce Očistec.[53]
Ocenění
Dne 28. října 2014 mu český prezident Miloš Zeman udělil Řád Bílého lva civilní skupiny I. třídy.
Odkazy
Reference
V tomto článku byl použit překlad textu z článku SMER – sociálna demokracia na slovenské Wikipedii.
 Fico získal diplom docenta v odbore bezpečnostnej služby. domov.sme.sk [online]. Petit Press a.s. [cit. 2023-05-23]. Dostupné online. (slovensky)
 UCL. Visit of Slovak Prime Minister. UCL News [online]. 2007-06-19 [cit. 2024-01-13]. Dostupné online. (anglicky)
 Fico si byt koupil za nižší než tržní cenu. Peníze jsem vysoudil od médií, tvrdí. iDNES.cz [online]. 2024-01-10 [cit. 2024-01-13]. Dostupné online.
 Fico koupil od stranického kolegy byt v Bratislavě se slevou 6 milionů - Novinky. www.novinky.cz [online]. 2024-01-11 [cit. 2024-01-13]. Dostupné online.
 BURČÍK, Matúš. Koľko stál Fica luxusný byt? Muňko mu ho predal za menej, ako ho sám kúpil. domov.sme.sk [online]. 2024-01-09 [cit. 2024-01-13]. Dostupné online. (slovensky)
 Ako Fico „vybýval“ Bašternákovu luxusnú rezidenciu: FOTO pred a po. plus7dni.pluska.sk [online]. 2020-03-03 [cit. 2024-01-13]. Dostupné online. (slovensky)
 Slovenský realitní magnát stráví pět let ve vězení. Přijde i o luxusní byt, který pronajímá Ficovi. iROZHLAS [online]. 2019-03-14 [cit. 2024-01-13]. Dostupné online.
 Slovenský parlament schválil vyslání jednotky do Iráku. Novinky.cz [online]. Borgis, 6. února 2003. Dostupné online.
 Slovenští vojáci se vrátili z Iráku. Český rozhlas [online]. 25. února 2007. Dostupné online.
 Slovenský premiér Fico: samostatnost Kosova byla chyba. Novinky.cz [online]. Borgis, 20. února 2008. Dostupné online.
 Slovensko není připraveno uznat samostatnost Kosova, řekl Fico. Deník.cz [online]. 2. dubna 2015. Dostupné online.
 iDnes: Slovenský prezident pověřil vytvořením vlády Radičovou. Fico přiznal prohru., 2010-06-23
 POLOCHOVÁ, Iveta. Slovenský premiér Fico chce být prezidentem. Oznámil kandidaturu. iDnes.cz [online]. 2013-12-18 [cit. 2013-12-18]. Dostupné online.
 prezident2014.statistics.sk [online]. [cit. 2014-03-30]. Dostupné v archivu pořízeném dne 2014-03-16.
 Výsledky 2. kola - Prezidentské voľby 2014. VýsledkyVolieb.sk [online]. [cit. 2019-10-11]. Dostupné v archivu pořízeném z originálu dne 2019-10-11.
 "Fico pospíchá do jádra Evropské unie". Novinky. 14. září 2017.
 Čína zrušila jednání s Ficem. Reaguje na setkání Kisky s dalajlámou. iDNES.cz [online]. 6. listopadu 2016. Dostupné online.
 Slovenská média: Vidíme konec Roberta Fica v přímém přenosu. Česká televize. 5. března 2018.
 Žádná mafie. Jána Kuciaka zavraždil amatér. Slovenský novinář zpochybňuje verzi policejního prezidenta. Seznam.cz. 6. března 2018.
 "Slovenský premiér Fico pokračuje v kritice prezidenta Kisky. Obvinil ho z destabilizace země a mocenských her". Hospodářské noviny. 6. března 2018.
 Slovakia rejects Crimea referendum. spectator.sme.sk. 24. března 2014. Dostupné online.
 Robert Fico: Idem do kampane, v ktorej ma chcú zatvoriť [online]. [cit. 2023-03-21]. Dostupné online. (slovensky)
 Fico kritizoval prezidenta Pavla za výrok o možném narušení vztahů. ČTK [online]. 24. září 2023. Dostupné online.
 Slováci už nechtějí pomáhat Ukrajině. Opozice boduje s mírovými hesly. Deník.cz [online]. 3. června 2023. Dostupné online.
 Odklon od Evropské unie a NATO směrem k Rusku. Fico cílí na extremistické voliče, říkají experti. iROHLAS.cz [online]. 2023-05-05 [cit. 2023-05-28]. Dostupné online.
 Fico, Pellegrini a Danko podepsali koaliční smlouvu o vzniku slovenské vlády. www.irozhlas.cz [online]. 2023-10-16 [cit. 2023-10-16]. Dostupné online.
 Tři slovenské strany v čele s Ficovým Smerem podepsaly koaliční dohodu. ct24.ceskatelevize.cz [online]. 2023-10-16 [cit. 2023-10-16]. Dostupné online.
 ČTK. Nová slovenská vláda premiéra Fica získala důvěru sněmovny. irozhlas.cz [online]. Český rozhlas, 2023-11-21 [cit. 2023-12-2O]. Dostupné online.
 Slovensko rozšíří zákaz dovozu zboží z Ukrajiny, rozhodla vláda. ČT24 [online]. Česká televize, 2023-11-29 [cit. 2023-12-25]. Dostupné online.
 ČTK. Slovensko chce zrušit prokuraturu, která vyšetřuje korupci. EU varuje před změnou trestního zákoníku. irozhlas.cz [online]. Český rozhlas, 2023-12-06 [cit. 2023-12-20]. Dostupné online.
 ČTK. Dražší cigarety, alkohol a speciální daň pro banky. Slovenská vláda navrhla nový konsolidační balíček. irozhlas.cz [online]. Český rozhlas, 2023-04-12 [cit. 2023-12-25]. Dostupné online.
 ČTK. Slovenský parlament schválil konsolidační balíček. Podle Fica má zajistit vyšší příjmy o 1,5 miliardy eur. irozhlas.cz [online]. Český rozhlas, 2023-12-19 [cit. 2023-12-25]. Dostupné online.
 NOVÁK, Ladislav. Snížení rozpočtu veřejnoprávní RTVS by mohlo ohrozit svobodu tisku, varují Reportéři bez hranic. irozhlas.cz [online]. Český rozhlas, 2023-12-14 [cit. 2023-12-25]. Dostupné online.
 Fico by chtěl odvolávat předsedu Ústavního soudu, už vyhlíží nového prezidenta - Novinky. www.novinky.cz [online]. 2024-03-05 [cit. 2024-03-05]. Dostupné online.
 Čaputová se kvůli novele trestního zákona obrátí na ústavní soud - Novinky. www.novinky.cz [online]. 2024-02-16 [cit. 2024-03-05]. Dostupné online.
 Slovenský ústavní soud pozastavil kontroverzní novelu trestního zákoníku - Novinky. www.novinky.cz [online]. 2024-02-29 [cit. 2024-03-05]. Dostupné online.
 „Kořeny války na Ukrajině jsou v roce 2014, kdy ukrajinští fašisté vraždili civilisty ruské národnosti,“ řekl Fico po pozastavení členství Smeru v PES. ct24.ceskatelevize.cz [online]. [cit. 2024-04-13]. Dostupné online.
 ČTK, iDNES cz. Fico se chystá na Ukrajinu. Není to suverénní země a do NATO nepatří, uvedl. iDNES.cz [online]. 2024-01-20 [cit. 2024-04-13]. Dostupné online.
 Robert Fico by blokoval případný návrh na vstup Ukrajiny do NATO - Novinky. www.novinky.cz [online]. 2024-02-24 [cit. 2024-03-02]. Dostupné online.
 Fico tvrdě zaútočil na západní podporu Ukrajiny a hájil Putina. ‚Je falešně démonizován,‘ prohlásil. iROZHLAS [online]. 2024-02-25 [cit. 2024-04-13]. Dostupné online.
 VHK. V Kyjevě žádná válka není, tvrdí Fico. „ Panuje tam úplně normální život“ - Echo24.cz. echo24.cz [online]. 2024-01-23 [cit. 2024-04-13]. Dostupné online.
 N, Tomáš Čorej, Denník. Fico otočil: Válka je nevyprovokovaná, Krym a Donbas patří Ukrajině. Na čem se shodl s ukrajinským premiérem. Deník N [online]. 2024-04-12 [cit. 2024-04-13]. Dostupné online.
 Fico odsoudil ruskou agresi na Ukrajině - Novinky. www.novinky.cz [online]. 2024-04-11 [cit. 2024-04-13]. Dostupné online.
 Unikly tajné nahrávky Roberta Fica. Matovič dští na jeho adresu vulgarismy. Seznam Zprávy [online]. Seznam.cz [cit. 2022-04-25]. Dostupné online.
 Slovenská policie má podle médií odposlechy ze schůzek Fica. Mají potvrzovat snahy ovlivňovat kauzy. iROZHLAS [online]. Český rozhlas [cit. 2022-04-25]. Dostupné online.
 Akce Očistec. Na Slovensku pozatýkali vedení policie z Ficovy éry. Seznam Zprávy [online]. Seznam.cz [cit. 2022-04-25]. Dostupné online.
 Slovenská policie zadržela exministra vnitra Kaliňáka, obvinila i Fica. Seznam Zprávy [online]. Seznam.cz, 2022-04-20 [cit. 2022-04-25]. Dostupné online.
 Slovenská policie obvinila Fica i Kaliňáka. Založili zločineckou skupinu, tvrdí. Deník.cz [online]. ČTK, 2022-04-20 [cit. 2022-04-25]. Dostupné online.
 DELIMAN, Michal. Ranný brífing: „Najlepší“ minister vnútra už sedí, Fico sa zberá za ním. SME.SK [online]. Petit Press, 2022-04-24 [cit. 2022-04-25]. Dostupné online. (slovensky)
 KELLÖOVÁ, Laura. Kauza Súmrak: Žilinkova prokuratúra zrušila obvinenie Ficovi aj Kaliňákovi. Aktuality.sk [online]. 2022-11-29 [cit. 2024-01-13]. Dostupné online. (slovensky)
 KINŠ, Viktor. Čo je paragraf 363, ktorým Žilinkova prokuratúra zastavuje vyšetrovania, a prečo je Boris Kollár ochotný povaliť preň vládu?. refresher.cz [online]. [cit. 2024-01-13]. Dostupné online.
 Slovenská prokuratura zrušila stíhání expremiéra Fica i exministra vnitra. Aktuálně.cz [online]. 2022-11-29 [cit. 2024-01-13]. Dostupné online.
 Strašná nemoc, řekl Fico kumpánům o covidu. Veřejně přitom hájí odmítače. iDNES.cz [online]. 2021-10-29. Dostupné online.
Související články
Vláda Roberta Fica – přehled vlád
Externí odkazy
Logo Wikimedia Commons Obrázky, zvuky či videa k tématu Robert Fico na Wikimedia Commons
 Osoba Robert Fico ve Wikicitátech
 Zpráva Premiér Fico nazval novináře idioty ve Wikizprávách
Stránka Roberta Fica na www.nrsr.sk
Ficův životopis na stránkách SMERu
Evropská rada
Premiéři Slovenska
První vláda Roberta Fica
Druhá vláda Roberta Fica
Třetí vláda Roberta Fica
Čtvrtá vláda Roberta Fica
Ministři obrany Slovenské republiky
Ministři spravedlnosti Slovenské republiky
Autoritní data Editovat na Wikidatech	
NKC: js20051003013 GND: 1073950670 ISNI: 0000 0001 0967 6494 LCCN: n96011805, no2016063530 NLP: a0000002873578 PLWABN: 9810591377605606 SNK: 168644 SUDOC: 202700801 VIAF: 43561737, 98146462668927771696  WorldCat Identities: lccn-n96011805
Portály: Lidé | Politika | Slovensko
Kategorie: Premiéři SlovenskaMinistři spravedlnosti SlovenskaPoslanci Národní rady Slovenské republikyČlenové KSČČlenové KSČ politicky aktivní po roce 1989Členové Strany demokratické leviceČlenové SMERu-SDNositelé Řádu Bílého lva I. třídyPředsedové slovenských politických stranKandidáti na prezidenta Slovenské republiky (2014)Narození 15. záříNarození v roce 1964Narození v TopoľčanechAbsolventi Právnické fakulty Univerzity Komenského v BratislavěPoslanci NR SR (2016–2020)
Stránka byla naposledy editována 13. 4. 2024 v 11:41.
Text je dostupný pod licencí Creative Commons Uveďte původ – Zachovejte licenci, případně za dalších podmínek. Podrobnosti naleznete na stránce Podmínky užití.
Ochrana osobních údajůO WikipediiVyloučení odpovědnostiKontaktujte WikipediiKodex chováníVývojářiStatistikyProhlášení o cookiesMobilní verze
Wikimedia FoundationPowered by MediaWiki
Přepnout omezenou šířku obsahu`;


vocab = new Vocab();
vocab.build(corpus);


// Define a trigram Bayesian network class
class TrigramBayesianNetwork {
  constructor() {
    this.network = {};
  }

  // Train the network using a list of numerical tokens
  update(a,b) {
    // Build the network by counting trigram transitions
   
     
      const nextNextToken = b;
      const trigram = a;

      if (!this.network[trigram]) {
        this.network[trigram] = {};
      }

      if (!this.network[trigram][nextNextToken]) {
        this.network[trigram][nextNextToken] = 1;
      } else {
        this.network[trigram][nextNextToken]++;
      }
    

    // Normalize the counts to probabilities
    Object.keys(this.network).forEach(trigram => {
      const transitions = this.network[trigram];
      const totalTransitions = Object.values(transitions).reduce((acc, count) => acc + count, 0);
      Object.keys(transitions).forEach(nextNextToken => {
        transitions[nextNextToken] /= totalTransitions;
      });
    });
  }

  // Generate the next token given two previous tokens
  generateNextToken(currentToken) {
    const trigram = currentToken;
    const transitions = this.network[trigram];
    if (!transitions) return null;

    // Randomly choose the next token based on the probabilities
    const randomValue = Math.random();
    let cumulativeProbability = 0;
    for (const [token, probability] of Object.entries(transitions)) {
      cumulativeProbability += probability;
      if (randomValue <= cumulativeProbability) {
        return token;
      }
    }

    return null; // Shouldn't reach here, but return null just in case
  }

  // Generate text autoregressively
  generateText(previousToken, length) {
    
    const generatedText = [previousToken];

    for (let i = 0; i < length - 1; i++) {
      const nextToken = this.generateNextToken(previousToken);
      if (!nextToken) break; // Stop if there are no more transitions
      generatedText.push(nextToken);
      previousToken = nextToken;
      
    }

    return generatedText;
  }

  // Save the model to a JSON file
  saveModel(filename) {
    const modelData = JSON.stringify(this.network, null, 2);
    fs.writeFileSync(filename, modelData);
    console.log(`Model saved to ${filename}`);
  }

  // Load the model from a JSON file
  loadModel(filename) {
    const modelData = fs.readFileSync(filename);
    this.network = JSON.parse(modelData);
    console.log(`Model loaded from ${filename}`);
  }
}




Q = [
    "Aké je hlavné mesto Francúzska?",
    "Kto je generálnym riaditeľom spoločnosti Tesla?",
    "Aký je najväčší planéta v našom slnečnej sústave?",
    "Ktorý hudobník je známy ako 'Král rock and rollu'?",
    "Aký je chemický symbol pre zlato?",
    "Kto bol prvým prezidentom Spojených štátov?",
    "Aká bola hlavnou príčinou druhej svetovej vojny?",
    "Kto postavil Veľkú pyramídu v Gíze?",
    "Kedy padol Rímsky cisárstvo?",
    "Kto viedol expedíciu, ktorá objavila Ameriku v roku 1492?",
    "Aký je proces, pri ktorom voda preniká cez rastlinu?",
    "Kto vynaliezol prvý počítač?",
    "Aký je najväčší žijúci druh jašterice?",
    "Koľko kostí má ľudské telo?",
    "Aké je najrýchlejšie suchozemské zviera?",
    "Kto napísal román 'Zabiť vtáča'?",
    "Aký je názov slávneho obrazu Leonarda da Vinciho?",
    "Kto skomponoval hudbu k opere 'Kúzelná flauta'?",
    "Aký je názov prvej knihy v sérii Harryho Pottera od J.K. Rowlingovej?",
    "Kto namaloval strop Sixtínskej kaplnky?",
    "Aký je najväčší púšť na svete?",
    "Ktorá rieka je najdlhšia v Južnej Amerike?",
    "Aký je najvyšší vrchol hory v Severnej Amerike?",
    "Ktoré mesto je známe ako 'Mesto kanálov'?",
    "Aký je najmenší štát na svete?",
    "Aké je odporúčané denné množstvo vitamínu C?",
    "Kto vyvinul prvú vakcínu proti osýpkam?",
    "Aký je najbežnejší typ rakoviny na svete?",
    "Koľko spánku potrebujú dospelí v priemere za noc?",
    "Aký je názov najväčšieho orgánu v ľudskom tele?",
    "Kto založil spoločnosť Microsoft Corporation?",
    "Aká je najväčšia burza na svete podľa trhovej kapitalizácie?",
    "Kto napísal 'Bohatstvo národov', základný text ekonómie?",
    "Aký je termín pre prvú verejnú ponuku akcií spoločnosti?",
    "Kto je považovaný za tvorcu moderných princípov riadenia?",
    "Kto hral postavu Lukea Skywalkera v pôvodnej trilógii Star Wars?",
    "Aké je hlavné mesto Austrálie?",
    "Kto je autorom diela 'Odysea'?",
    "Aký je druhá odmocnina z 64?",
    "Kto vyhral prvú Nobelovu cenu za literatúru?",
    "Aký je chemický symbol pre kyslík?",
    "Kto bola prvou ženou, ktorá získala Nobelovu cenu?",
    "Aká bola hlavnou príčinou Francúzskej revolúcie?",
    "Kto objavil penicilín?",
    "Kedy začala prvá svetová vojna?",
    "Kto bol druhým prezidentom Spojených štátov?",
    "Aký je proces, pri ktorom sa rastliny menia svetelnú energiu na chemickú energiu?",
    "Kto vynaliezol žiarovku?",
    "Aký je najväčší cicaviec na Zemi?",
    "Koľko sŕdc má chobotnica?",
    "Aký je najrýchlejší vták na svete?",
    "Kto režíroval film 'Základňa'?",
    "Aký je názov slávnej sochy Michelangela?",
    "Kto skomponoval hudbu k baletu 'Jazero labutí'?",
    "Aký je názov druhej knihy v sérii Harryho Pottera od J.K. Rowlingovej?",
    "Kto namaloval 'Hviezdnatú noc'?",
    "Aký je najväčší oceán na svete?",
    "Ktorá rieka je najdlhšia v Afrike?",
    "Aký je najvyšší vodopád na svete?",
    "Ktoré mesto je známe ako 'Mesto svetiel'?",
    "Aký je najväčší planéta v našom slnečnej sústave?",
    "Aké je odporúčané denné množstvo vody?",
    "Kto vyvinul vakcínu proti obrne?",
    "Aký je najbežnejší krvný typ?",
    "Koľko komôr má ľudské srdce?",
    "Kto založil spoločnosť Apple Inc.?",
    "Aká je najväčšia technologická spoločnosť na svete podľa tržieb?",
    "Kto napísal 'Manifest komunistickej strany'?",
    "Aký je termín pre prvú verejnú ponuku akcií spoločnosti?",
    "Kto je pripisovaný s rozvojom teórie relativity?",
    "Kto hral Jamesa Bonda v prvom bondovskom filme 'Dr. No'?",
    "Aké je hlavné mesto Japonska?",
    "Kto je autorom diela '1984'?",
    "Aký je vzorec vody?",
    "Kto vyhral prvú cenu Akadémie za najlepšieho herca?",
    "Aký je chemický symbol pre vodík?",
    "Kto bol prvý človek, ktorý kročil na Mesiac?",
    "Aká je hlavná príčina zmeny klímy?",
    "Kto objavil gravitáciu?",
    "Kedy skončila druhá svetová vojna?",
    "Kto bol tretím prezidentom Spojených štátov?",
    "Aký je proces, pri ktorom zelené rastliny vyrábajú potravu zo slnečného svetla?",
    "Kto vynaliezol telefón?",
    "Aký je najväčší plaz na Zemi?",
    "Koľko zubov má dospelý človek?",
    "Aký je najrýchlejší ryba na svete?",
    "Kto napísal divadelnú hru 'Romeo a Júlia'?",
    "Aký je názov slávneho obrazu od Edvarda Muncha?",
    "Kto skomponoval hudbu k symfónii 'Nový svet'?",
    "Aký je názov tretej knihy v sérii Harryho Pottera od J.K. Rowlingovej?",
    "Kto vytesal sochu Davida?",
    "Aký je najväčší kontinent na svete?",
    "Ktorá rieka je najdlhšia v Severnej Amerike?",
    "Aký je najvyšší vrchol na svete?",
    "Ktoré mesto je známe ako 'Večné mesto'?",
    "Aký je najmenší planéta v našom slnečnej sústave?",
    "Aké je odporúčané denné množstvo bielkovín?",
    "Kto vyvinul vakcínu proti osýpkam?",
    "Aká je najbežnejšia farba očí?",
    "Koľko pľúc má človek?",
    "Kto založil spoločnosť Amazon.com?",
    "Aká je najväčšia e-commerce spoločnosť na svete podľa tržieb?",
    "Kto napísal 'Teóriu evolúcie'?",
    "Aký je termín pre predaj novovydaných akcií spoločnosti existujúcim akcionárom?",
    "Kto je považovaný za tvorcu teórie prirodzeného výberu?",
    "Kto hral Iron Mana vo filme Marvel Cinematic Universe?"
]

         Q.forEach(q => {
              
               const postData = {
                "prompt": `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nSi nápomocný AI menom K8 stále rád odpovedáš na otázky po slovensky<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n${q}<|eot_id|>\n`,
                "model": "meta/meta-llama-3-70b-instruct",
                "systemPrompt": "You are a helpful assistant.",
                "temperature": 0.75,
                "topP": 0.9,
                "maxTokens": 200,
                "image": null,
                "audio": null
               }

               axios.post(' https://www.llama2.ai/api', postData)
  .then(response => {
    // Handle successful response

   
    

  TT = vocab.tokenize(response.data)
  let i = 0;
for(i; i<TT.length;i++){
  G = TT[i]*Math.cos(i)
    
if(TT[i+1]===undefined){break;}
    trigramBayesianNetwork.update(TT[i],TT[i+1]);}
    //const startingWords = vocab.tokenize('Na');
    const generatedText = trigramBayesianNetwork.generateText(TT[i], 50);
    console.log("Generated",generatedText)
    console.log("Generated text:", vocab.detokenize(generatedText));

// Save the model to a file
trigramBayesianNetwork.saveModel('trigram_model.json')

// Use the loaded model to generate text




   
  })
  .catch(error => {
    // Handle error
    console.error('Error:', error);
  });
})  


                   
       




