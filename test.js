function padArray(arr, targetLength, padValue) {
  // Calculate the number of elements to pad
  const padAmount = Math.max(0, targetLength - arr.length);

  // Create a new array with the padding values
  const paddedArray = [...arr, ...Array(padAmount).fill(padValue)];

  return paddedArray;
}

class BPETokenizer {
  constructor() {
      this.vocab = new Set();
      this.tokenToID = new Map();
      this.maxSubwordLength = 4;
      this.subwordFrequencies = new Map();

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
      
      tokens.encoded = tokenIDs
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


  const words = text.match(/'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu)
  
  

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


clusterTokens(similarityThreshold) {
  const coOccurrenceMatrix = this.buildCoOccurrenceMatrix(corpus);
  const tokenIDs = Array.from(this.vocab).map(subword => this.tokenToID.get(subword));
  const tfidfMatrix = this.calculateTFIDF(coOccurrenceMatrix, tokenIDs);
  const similarities = this.calculateSimilarities(tfidfMatrix);
  const clusters = this.hierarchicalClustering(similarities, similarityThreshold);
  return clusters;
}

buildCoOccurrenceMatrix(corpus) {
  const coOccurrenceMatrix = new Map();
  for (const word of corpus) {
      const subwords = this.getSubwords(word);
      for (const subword of subwords) {
          if (!coOccurrenceMatrix.has(subword)) {
              coOccurrenceMatrix.set(subword, new Map());
          }
          for (const otherSubword of subwords) {
              if (subword !== otherSubword) {
                  const count = coOccurrenceMatrix.get(subword).get(otherSubword) || 0;
                  coOccurrenceMatrix.get(subword).set(otherSubword, count + 1);
              }
          }
      }
  }
  return coOccurrenceMatrix;
}

calculateTFIDF(coOccurrenceMatrix, tokenIDs) {
  const tfidfMatrix = [];
  for (const tokenID of tokenIDs) {
      const tfidfVector = [];
      const token = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === tokenID);
      const coOccurrences = coOccurrenceMatrix.get(token);
      let docFreq = 0;
      for (const subword of coOccurrences.keys()) {
          docFreq += 1;
      }
      for (const otherTokenID of tokenIDs) {
          const otherToken = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === otherTokenID);
          const tf = coOccurrences.get(otherToken) || 0;
          const idf = Math.log(corpus.length / docFreq);
          tfidfVector.push(tf * idf);
      }
      tfidfMatrix.push(tfidfVector);
  }
  return tfidfMatrix;
}

calculateSimilarities(tfidfMatrix) {
  const similarities = [];
  for (let i = 0; i < tfidfMatrix.length; i++) {
      const rowSimilarities = [];
      for (let j = 0; j < tfidfMatrix.length; j++) {
          const similarity = this.cosineSimilarity(tfidfMatrix[i], tfidfMatrix[j]);
          rowSimilarities.push(similarity);
      }
      similarities.push(rowSimilarities);
  }
  return similarities;
}

cosineSimilarity(vec1, vec2) {
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

hierarchicalClustering(similarities, similarityThreshold) {
  // Hierarchical clustering implementation (you can use an existing library or implement it manually)
  // For demonstration purposes, I'll provide a basic example using a threshold-based approach
  const clusters = [];
  for (let i = 0; i < similarities.length; i++) {
      const cluster = [];
      for (let j = i + 1; j < similarities.length; j++) {
          if (similarities[i][j] >= similarityThreshold) {
              cluster.push(j);
          }
      }
      if (cluster.length > 0) {
          cluster.push(i); // Include the current token in the cluster
          clusters.push(cluster);
      }
  }
  return clusters;
}

 
}

// Example usage:
const corpusText = `
Main menu
 
WikipediaThe Free Encyclopedia

Search
Create account
Log in

Personal tools
Contents  hide
(Top)
Gameplay
Toggle Gameplay subsection
Story and dialogue
Development
Toggle Development subsection
Conception
Design
Writing
Music
Release
Toggle Release subsection
Sales
Reception
Toggle Reception subsection
Accolades
Downloadable content and future
References
Toggle References subsection
Citations
Works cited
External links
Katana Zero

13 languages
Article
Talk
Read
View source
View history

Tools
 Featured article
Page semi-protected
From Wikipedia, the free encyclopedia
Katana Zero
An illustration of a katana-wielding assassin attacking several thugs brandishing firearms, with the words "KATANA ZERO" in the center. The image is coloredc using neon blues, yellows, pinks, and purples.
Developer(s)	Askiisoft
Publisher(s)	Devolver Digital
Designer(s)	Justin Stander
Programmer(s)	Justin Stander
Writer(s)	
Eric Shumaker
Justin Stander
Composer(s)	
Thijs Lodewijk
Bill Kiley
Engine	GameMaker Studio 2
Platform(s)	
macOS
Nintendo Switch
Windows
Xbox One
Android
iOS
Release	
macOS, Switch, Windows
April 18, 2019
Xbox One
October 15, 2020
Android, iOS
TBA
Genre(s)	Platform, hack and slash
Mode(s)	Single-player
Katana Zero is a 2019 platform game created by the indie developer Justin Stander. Set in a dystopian metropolis, the neo-noir storyline follows Subject Zero, a katana-wielding assassin with amnesia who can slow down time and predict the future. Zero unravels his past while completing assassination contracts. Katana Zero features side-scrolling hack-and-slash gameplay in which the player attempts to kill all enemies in a level without being hit, using Zero's abilities to manipulate time, dodge attacks, and take advantage of environmental hazards. In between levels, the story is told in sequences where the player converses with non-player characters through dialogue trees.
Stander began working on Katana Zero in 2013. He had previously developed freeware games, such as Tower of Heaven (2009), and conceived Katana Zero as his first commercial game. Using GameMaker Studio 2, Stander sought to make a difficult story-driven game that did not force the player to wait through dialogue and cutscenes. He focused on attention to detail and looked to films such as Sin City (2005) and John Wick (2014) for story inspiration. The development was prolonged and Stander worked mostly alone, although he recruited artists to design the visuals as well as musicians Bill Kiley and Thijs "LudoWic" Lodewijk to compose the synthwave soundtrack.
Katana Zero was published by Devolver Digital for macOS, the Nintendo Switch, and Windows on April 18, 2019. It sold 500,000 copies in less than a year and received positive reviews. Critics praised the gameplay—which they favorably compared to Devolver's Hotline Miami (2012)—and the visuals, writing, and music. The story divided reviewers and the unresolved ending was criticized. Several critics cited Katana Zero as one of the best games of 2019 and it was nominated for numerous year-end accolades. A port for the Xbox One was released in 2020, while Android and iOS versions and downloadable content are in development. Stander intends to continue the fictional universe in future games.
Gameplay
A 21-second GIF that shows the player character in a purple room, being pursued by enemies before using a slow motion effect to dodge and kill them. The heads-up display at the top shows (from left to right) the amount of slow motion they can use, the number of deaths, the time limit, the total elapsed time, and the player's on-hand weapons.
A GIF illustrating many of the core game mechanics of Katana Zero, including its use of bullet time and one-hit-kill gameplay
Katana Zero is a 2D platform and hack and slash game presented from a side-scrolling perspective.[1][2] Controlling the player character, the katana-wielding assassin Subject Zero,[3] the single player completes assassination contracts for a psychiatrist. Zero can run, jump, wall kick, pick up and throw items, attack using his katana, and dodge.[4] Zero's ability to slow down time and predict the future allows the player to activate a slow motion effect, enabling them to predict enemy movement easier, although use is limited by a meter that gradually refills.[5][6] The game features eleven levels,[7] which use Zero's precognition as a framing device; the player's attempts to complete each level are presented as possible scenarios Zero has foreseen.[1][5]
Levels are split into several rooms, and the player must kill every enemy in a room using their sword, throwable objects, such as lamps and pots, or environmental hazards, such as lasers.[4][7] Aside from occasional bosses, each enemy dies in a single hit.[1] Certain levels feature unique game mechanics, such as a stealth mission in a nightclub,[8] a motorcycle chase,[9] and an alternate player character.[4] Any damage results in instant death for Zero, requiring the player to restart from the most recent checkpoint.[7] Katana Zero has been frequently compared to Hotline Miami (2012),[8][10] as both feature levels filled with enemies, one-hit kills, and require players to determine their chosen route strategically.[11] Outside of the main game, there are two additional game modes: hard mode features more difficult levels with new enemy varieties, reworked bosses, and additional challenges; and speedrun mode challenges the player to complete every level in the fastest time possible, with the options to modify enemy behavior and skip cutscenes.[12]
Story and dialogue
Katana Zero follows a neo-noir-style storyline, with psychological horror and black comedy elements,[13] set in a dystopian metropolis after a war.[2][5] Subject Zero is an amnesiac veteran with precognitive abilities.[14][15] He assassinates drug dealers for his psychiatrist,[8] who acts as his handler.[16] The news media ascribes these killings to a serial killer known as the Dragon.[13] Zero experiences recurring nightmares of a child, who he identifies as himself, in a hut. A scientist runs in, warns the child to hide, and is shot moments later by a soldier.[17] He discusses his nightmares with the psychiatrist, who supplies him with a drug as treatment.[18] Zero also befriends a young girl living next door to his apartment,[7][13] and he becomes attached to her.[15]
In between levels, the player converses with non-player characters (NPCs), such as the psychiatrist, the girl, and a Russian psychopath antagonist named V, who admires Zero's lethality.[5][7][13] In a real-time dialogue tree system, the player chooses responses during conversations and can interrupt an NPC's dialogue at any time.[16] Their decisions determine how much exposition is presented and how Zero is characterized;[1] for example, Zero comes across as rude if the player repeatedly interrupts.[19] Although they do not change the overall plot, players' dialogue choices can affect certain events, and one boss fight can only be activated by making specific decisions.[16]
Zero and the psychiatrist's relationship becomes strained as the psychiatrist grows increasingly disagreeable and Zero suspects he is withholding information about the assassinations.[7][20] After the Dragon, a separate swordsman with clairvoyant abilities similar to Zero's, dismembers and abducts V,[21] Zero learns about his own past as a supersoldier and that the drug he had been taking, Chronos, both gave him his abilities[22] and causes users in withdrawal to become trapped within their minds.[23] Zero, tired of being manipulated, kills the psychiatrist. The girl goes missing[24] and the story ends on a cliffhanger.[13][15] A flashback reveals Zero's nightmare is a memory from the war, that he is the soldier and not the child, and that the Dragon was his comrade.[25]
Development
Conception
Katana Zero was developed over six years by the indie game creator Justin Stander under the studio name Askiisoft. It was Stander's first commercial game; his previous projects, such as Tower of Heaven (2009), had been smaller freeware games. After seeing the success of Terry Cavanagh's VVVVVV (2010), Stander concluded audiences only pay attention to indie games if they are being sold. Cavanagh, like Stander, had started off making freeware games, but none were as successful as VVVVVV. Katana Zero originated from Stander's desires to create a larger project that could be sold commercially and tell a story.[10] He began working on it in 2013 as a hobby during his sophomore year at McGill University.[26] He used the GameMaker Studio 2 game engine and spent the first two years building simple prototypes.[27][28] The game was a means of expression for Stander outside schoolwork and he spent most of his time at college developing it.[26]
After Stander graduated in 2015,[26] he developed Katana Zero full time.[10] He worked on multiple projects alongside it as a precaution since he felt the chances of success were slim.[29] The total budget was US$60,000, which Stander noted was quite small for a game of Katana Zero's scope.[10] He stated: "Most of it was just not paying myself at all and cutting down costs in my own life to do nothing but work on the game."[10] Stander worked largely on his own, although he recruited help for the art and music.[30] The game was initially developed for personal computers (macOS and Windows).[4][31] Stander decided to develop a Nintendo Switch version immediately after the system was unveiled because he saw it as a good console for indie games.[30] GameMaker made it easy to port Katana Zero and the long development meant it was already well optimized.[30]
Design
One of Stander's goals was to make killing feel exciting and satisfying. He considered many modern games too forgiving, with enemies less powerful than the player. Finding it difficult to die in games such as Payday: The Heist (2011), Stander decided it would be easy to die in Katana Zero.[10] He wanted the game to be difficult but fair and for the player to recognize and take responsibility for their mistakes.[32] He continued his style of design—short levels filled with instant-death scenarios—from Tower of Heaven and Pause Ahead (2013).[33] The one-hit-kill gameplay was frequently compared to another Devolver Digital-published game, Hotline Miami. Stander said he only played Hotline Miami once and did not remember its gameplay, but acknowledged it may have subconsciously influenced him.[10] He named Samurai Gunn (2013), which he felt used one-hit kills effectively, as the bigger influence.[10]
Stander wanted Katana Zero to feel cinematic and sought to subvert expectations: "As soon as you think you understand how this game is going to play out, then I just try to completely shift it on you... as soon as [you're] comfortable in [something], I try to shift things up. And I do that several times throughout the game. I really mess with the player."[32] To maintain variety, he incorporated many enemy types, environmental traps, alternate level pathways, and set pieces. A minecart pathway inspired by Indiana Jones and the Temple of Doom (1984) took over a month to create.[32] Stander looked to indie games that feature "tight, fast-paced, instant death combat" for inspiration,[34] such as Trilby: The Art of Theft (2009) and Gunpoint (2013).[33]
The art style was inspired by the neon lighting aesthetics associated with the 1980s.[34] Recruiting artists proved challenging; Stander called himself "a terrible artist", and for two years no artists worked on the game.[10] He found artists through the online independent developer community TIGSource, but said it was difficult to recruit high-quality pixel artists who would commit to the project. Many would only work on it for weeks or days before quitting for various reasons, such as other commitments or feeling their style did not match.[10] Stander used the neon lighting to blend the artists' different styles.[10] He considered being able to get an artist team to finish the game mere chance and credited the artists with motivating him to finish.[10][26]
Stander focused on attention to detail and said adding a single mechanic, such as a gun turret, would require him to alter many different systems, such as the lighting, to maintain cohesion.[26] Production value was important, as Stander wanted the engine to feel flawless.[32] He said this meant "no bugs. Everything needs to feel like an extension of the player. When you play it, you should always feel like this is exactly what I wanted to do and that is what the character did."[32] When developing small freeware games, he could scrutinize minute details, which would result in him spending a year to make a short, 20–30 minute game.[10] Adapting this mentality to a full-length commercial game contributed to the prolonged development; Stander originally expected development to last a year or two.[10][26] His focus on particulars demoralized him, to the point he made little progress for a year.[26]
Writing
Stander said telling a story was a large part of his motivation to develop Katana Zero, wanting to celebrate his favorite tropes and provide his own spin on them.[10] The script is credited to Stander and Eric Shumaker, with additional writing by Sterling Nathaniel Brown and Ian Goldsmith Rooney.[25] As Stander developed Katana Zero focusing on one element at a time, he only had a basic plot summary by the time he finished outlining all the levels. Elements Stander conceived early on included a protagonist who was "sort of trapped in their situation, because those always make for good main characters in action games," and a disagreeable psychiatrist whom players would dislike.[16]
Katana Zero is a pastiche of films Stander enjoyed.[33] He was inspired by the Eastern culture of samurai cinema and wanted a vulnerable yet lethal protagonist similar to those from Korean revenge thrillers.[34][10] Film influences included Oldboy (2003) Sin City (2005), Drive (2011), and John Wick for their "invincible-yet-human" protagonists and "stylistic violence set over a dark, grimy, neon coated setting,"[33] as well as Seven Samurai (1954) and the films of Quentin Tarantino.[30][34] The story structure was inspired by Hotline Miami, in which the player character is directed to kill by mysterious phone calls, and its themes include drug addiction and mental health. Stander hesitated to deal with such topics as they had never affected him, but after some research, felt he could treat them respectfully.[16] Stander said the script was rewritten around 30 times.[26]
The player character converses with a non-player psychiatrist character in front of a fireplace. A speech bubble from the psychiatrist reads: "So what if it comes from a drug? YOU HOLD THE POWER OF A GOD! YOU SEE THE FUTURE! How DARE you squander this gift! Now SIT. STILL." Beneath the characters, there is a timer and two dialogue responses the player can choose: "Okay" and "No".
An example of Katana Zero's dialogue tree system. Stander conceived the system for a role-playing video game that he never developed and used graphical effects to emphasize major decisions.
Stander allowed the player to interrupt any line of spoken dialogue because many of the action games he grew up playing "would grind to a halt as the protagonist and the second banana argued about politics, or when the villain deigned to deliver a winding monologue."[16] He considered this one of the biggest problems with cutscenes, as they strip players of their agency.[35] Additionally, Stander felt an assassin like Zero would not wait to listen to a villain justify their schemes. He was conscious speedrunners would skip all the dialogue and included in-game consequences for interrupting constantly to create a sense of realism.[16] He adopted a show-don't-tell approach to convey as much of the story, themes, and characters as possible through just visuals. A level in which the player controls the Dragon was intended to show the character's lethality without words.[16]
The dialogue tree system originated from Stander's concept for a role-playing video game (RPG) that centered on limitlessness, allowing the player to interact with any object and fight any NPC.[35] He wanted every action to have repercussions, which he likened to the Grand Theft Auto games. Although he never developed the RPG, Stander reincorporated the dialogue system in Katana Zero to keep the pacing consistent. Difficulty arose from finding the right timing between interrupting and responding. Playtesters would choose the wrong response if they interrupted too late, or unintentionally interrupt by taking too much time to respond.[35] Stander resolved the problem using "coyote time", a trick in game development in which developers provide the player a brief interval to make their decision even if the on-screen window of opportunity has passed.[35]
Stander kept the stakes for each dialogue choice minimal to reduce tension and make choosing a response feel natural. He noted "even the ones that seem big will peter out or resolve themselves," similar to Telltale Games' The Walking Dead series, such as an instance in which a decision will cause the player to temporarily lose their sword.[35] However, he said "the way the story is told is entirely dependent on [the player's] choices," noting worldbuilding and character relationships can change depending on the dialogue options. For example, Zero's relationship with the psychiatrist suffers if the player ignores his orders, which can lead to alternate story paths in which Zero learns information he was not supposed to.[32] Stander used graphical effects to emphasize major decisions, such as moving or colored text and character animations. He accomplished this by programming the script to affect other areas of the game upon reaching certain points. His initial intention was only to add polish, but he began to experiment and added elements such as distortion effects and screen-shaking.[35]
Music
Katana Zero's synthwave score, which blends Chicago house, electronic music, and synth-pop,[36] was composed by Bill Kiley and Thijs "LudoWic" Lodewijk, with additional music by Stander.[25] Kiley had collaborated with Stander in the past and Stander recruited him to work on Katana Zero at the beginning of development.[36] Lodewijk, who had never composed a video game, became involved in 2015, after Stander found his YouTube channel and recruited him to write a single track. Stander then asked if he could use music from Lodewijk's "jams" (improvised recording sessions Lodewijk had uploaded to his channel);[37] the jam that first drew Stander's attention to Lodewijk, "Jam #12," was used as the boss theme.[38] Lodewijk composed a track specifically for Stander after joining the project, but Stander told him to compose as he normally would.[39]
Kiley and Lodewijk looked to 1980s electronic music for inspiration since the game's themes—including drug use and the effect of war on a nation's spirit—were relevant in the 1980s in the aftermath of the Vietnam War. They originally worked separately, as Stander wanted Kiley to write muted music for story-driven scenes and Lodewijk to write energetic music for the main levels.[36] According to Kiley, "As the project progressed we gleefully broke this rule and ended up writing music for levels and story scenes all over the place."[36] Kiley and Lodewijk attempted to reflect Zero's changing psyche and moods in their music, such as when Zero experiences Chronos withdrawal and snorts cocaine in a limousine.[36] Kiley drew influence from the work of Gary Numan, Yellow Magic Orchestra, and Vangelis,[36] and sought to evoke the feelings of 1980s action films.[40]
Stander sent Kiley and Lodewijk screenshots, concept art, and notes describing the atmosphere he was aiming for, occasionally alongside an existing piece of music for reference.[37][40] Lodewijk explained they would send their initial composition to Stander, leading to a series of exchanges that resulted in the final track.[37] Some tracks ended up in different levels than intended, and for certain levels and dream sequences, Lodewijk wrote two tracks for Stander to blend in-game.[41] Lodewijk composed most tracks in a single take, "[letting the music] go and sort of adjust[ing] it as it [went] along."[42] He drew inspiration from Nine Inch Nails, as he and Stander were fans of their dark and industrial tone.[39] Lodewijk did not set out to make dark music, but he noted his tracks were often somber, even when he attempted to compose happy music.[42]
Kiley said he composed using the same monitors Richard "Disasterpeace" Vreeland used to compose Fez (2012),[43] while Lodewijk used vintage synthesizers and drum machines such as the Akai MPC.[42][36] While he did edit some tracks using his computer, Lodewijk described his process as "old-fashioned" and estimated 95% of his music was composed using vintage synths.[42] Stander attempted to synchronize the music with the gameplay by having Zero turn on a Walkman at the beginning of each level, changing the music when Zero takes his earbuds out to talk to an NPC, and slowing down the music when the player uses Chronos.[32]
Release
Katana Zero made its first public appearance at PAX West in Seattle in September 2015.[44] Adult Swim Games obtained the publication rights, and a teaser trailer was released in December.[45] The game was scheduled for a late 2016 release,[46] but it was delayed to 2017 and eventually to March 2019.[47][48] Stander announced he had "amicably parted ways" with Adult Swim in December 2018,[48] and revealed Devolver Digital had acquired the publication rights the following month.[49] Devolver helped Stander localize the game, translating it to ten languages. Stander was impressed by Devolver's software testing process for catching bugs he did not notice.[30]
Katana Zero was released on April 18, 2019,[50] as a downloadable game on GOG.com, Humble Bundle, Nintendo eShop, and Steam.[51][52] The Switch version was temporarily banned in Australia after it was refused classification by the International Age Rating Coalition,[53] due to its depiction of graphic violence and drug use.[54] Devolver Digital resubmitted the game to the Australian Classification Board, which cleared it for a May release with an R18+ rating.[55] Devolver released an Xbox One version on October 15, 2020, offered to Xbox Game Pass subscribers,[56] and an Amazon Luna version on December 9, 2021.[57] A PlayStation 4 version was rated by the Entertainment Software Rating Board in March 2021,[58] but has not been released as of September 2022.[59] Android and iOS ports were announced in November 2023 as being released as part of the Netflix subscription service.[60]
Sales
Katana Zero was Devolver's most-preordered Switch game, sold over 100,000 copies within a week of release, and became Devolver's second-fastest-selling Switch game behind Enter the Gungeon (2017).[61] It was the second-bestselling eShop game during the month following its release behind the Switch version of Cuphead (2017), which was released on the same day.[10] Katana Zero sold 500,000 copies in less than a year and generated US$5 million in revenue. In contrast, the average indie game generates around US$16,000.[62] Stander said Katana Zero was most successful on Switch and Steam; sales were originally strongest on Switch, but the Steam version gradually sold more since it went on sale often.[10]
Reception
Reception
Aggregate scores
Aggregator	Score
Metacritic	83/100[63][64]
OpenCritic	87%[65]
Review scores
Publication	Score
Destructoid	9/10[9]
Game Informer	7.75/10[66]
GameSpot	8/10[2]
IGN	8.7/10[5]
Nintendo Life	9/10[6]
Nintendo World Report	9/10[67]
PC Gamer (US)	79%[7]
USgamer	4.5/5[23]
Katana Zero received "generally favorable reviews" according to the review aggregate website Metacritic,[63][64] and a "Mighty" approval rating from OpenCritic.[65] Critics considered Katana Zero stylish and well designed. IGN said it "refines the tried and true one-hit-kill formula in a manner that makes it feels fresh, exciting, and innovative",[5] while Nintendo Life and Nintendo World Report considered it a standout in the Nintendo eShop library.[6][67] Destructoid's reviewer said Katana Zero was "bleak, beautiful, bloody, and brilliant" and changed how he viewed video games.[9]
Critics praised the 16-bit visuals. They enjoyed the retro VHS aesthetic and visual effects[6][9][67][23]—though some wrote their intensity could induce headaches[1][67]—as well as the amount of detail in the sprites and animations.[2][6][5] IGN found the lighting effects impressive and said they worked with the "slick neon aesthetic and fantastic sprite work" to give the game personality.[5] GameSpot appreciated that Katana Zero did not use a retro aesthetic simply for nostalgia,[2] and alongside Polygon found the detailed sprite work and smooth animations added emotional weight.[2][68]
The soundtrack was acclaimed.[5][6][9] Destructoid called it an "audial delight"[9] and Nintendo Life's reviewer said the music overshadowed the rest of the game at points. He felt the soundtrack had a clear focus but remained "willing to experiment with obscure genres" and complimented the composers for doing something original in contrast to the "cliché" chiptune style prevalent in indie games.[6] IGN said the soundtrack was excellent and fitting,[5] and PC Gamer liked how it was contextualized in the world.[7]
Although reviewers praised the writing,[5][6] they were divided on the story. Destructoid noted it was difficult to discuss since much is left to the player's interpretation.[9] Shacknews praised the plot twists,[1] and GameSpot said the story did a good job balancing graphic violence with "delicately quiet character moments and some heartfelt relationships".[2] Some commended how the story provided context for the slow-motion game mechanic,[4][68] and others thought it had heart.[2][4][68] However, PC Gamer found the plot generic dystopian fiction with stock characters,[7] while Game Informer felt it had interesting ideas but "most of them just cryptically meander without reaching any crescendo".[66] Multiple reviewers disliked the ending, which they called abrupt, and felt teased story moments lacked payoffs.[2][5][66][23] GameSpot found this problematic since a sequel was not guaranteed.[2]
The dialogue system was considered creative and innovative.[1][2][5][67] Reviewers said the branching dialogue paths and alternate story scenarios added replay value.[5][1][67] The graphical effects used to emphasize dialogue were praised for adding emotional weight to conversations.[4][2] Destructoid and Nintendo Life said the dialogue system helped make the story interesting,[6][9] and Polygon wrote it helped the player build an emotional connection with Zero.[68] Minor criticism came from IGN, which said many story deviations felt superficial, leading to later choices feeling inconsistent with those made earlier. Nonetheless, IGN said the dialogue system was entertaining and encouraged multiple playthroughs.[5]
Critics enjoyed the fast, fluid gameplay, which they frequently compared to Hotline Miami.[1][4][6][7][23] Rock, Paper, Shotgun and IGN said the influences were obvious, but this was not a problem.[5][4] IGN said the controls were "empoweringly flexible"[5] and Nintendo Life said it felt great to learn them.[6] Reviewers frequently compared Katana Zero to a puzzle game,[2][67][23][66] requiring the player to strategize and plan. They said this made completing each level feel satisfying,[1][23][66] with Game Informer considering the trial-and-error process of polishing movements "the most entertaining part of [the game]".[66] Conversely, Polygon felt the gameplay, while good, did not live up to the presentation, and that the puzzle-like design made the game feel limited.[68] The short length was noted,[5][6][23] but IGN and Nintendo Life said it worked in the game's favor since it meant there was no filler.[5][6]
Reviewers said the player character's abilities were static and limited.[6][23][66] While Game Informer said this was a problem,[66] USgamer found the game was "all about playing with those scant toys",[23] and PC Gamer felt they were "adaptable enough to make the combat encounters varied".[7] GameSpot,[2] USgamer,[23] Destructoid,[9] and IGN felt the gameplay stayed interesting with its set pieces and variety of enemy types,[5] though Game Informer disagreed.[66] IGN's reviewer said the Chronos slow motion mechanic was his favorite element since it was powerful but still limited,[5] and Nintendo Life described the process of balancing Chronos use throughout the levels as exciting.[6] Destructoid praised how enemies never reset to their original position after deaths,[9] although Game Informer considered this an annoyance.[66]
Accolades
Katana Zero was among the top 50 highest-rated games on Metacritic in 2019,[69] and was named one of the best games of 2019 by USgamer (#1),[70] Thrillist (#21),[71] and Red Bull.[72] IGN nominated it for "Best Action Game" and "Best Video Game Music/Soundtrack" during its Best of 2019 Awards.[73][74] For his work on the game, Stander was included in Forbes' 2020 30 Under 30 list.[62]
Year	Award	Category	Result	Ref(s).
2019	2019 SXSW Gaming Awards	Gamer's Voice: Video Game	Nominated	[75]
The Game Awards 2019	Best Independent Game	Nominated	[76]
2020	Independent Games Festival Awards	Excellence in Design	Nominated	[77]
2020 SXSW Gaming Awards	Most Promising New Intellectual Property	Nominated	[78]
16th British Academy Games Awards	Debut Game	Nominated	[79]
Downloadable content and future
On April 25, 2019, a week after Katana Zero's release, Stander announced he was working on free downloadable content (DLC).[80] He wanted the DLC's quality to be on par with the main game's, and its size expanded considerably during development.[81] It was three times its originally-planned size by February 2020,[82] and six times by March 2021.[81] However, the expansions did not change Stander's plans to release the DLC for free.[81] The DLC will be slightly more than half the size of the base game and will introduce new game mechanics, enemies, and story elements. Stander described it as "more like Katana 1.5" than DLC.[81] He said the DLC will resolve some plot threads and continue the worldbuilding, but will not complete the story.[29] Stander plans to continue the story beyond the DLC and has its conclusion planned. In May 2020, he said some of his future games would connect to its fictional universe.[29]
References
Citations
 Mejia, Ozzie (April 18, 2019). "Katana Zero review: The cutting edge". Shacknews. Archived from the original on February 8, 2022. Retrieved February 13, 2022.
 Barbosa, Alessandro (April 19, 2019). "Katana Zero Review - Slow-Motion For Me". GameSpot. Archived from the original on February 8, 2022. Retrieved February 13, 2022.
 Gault, Matthew (June 26, 2019). "The Best Video Games of 2019 (So Far)". Time. Archived from the original on October 26, 2021. Retrieved October 16, 2022.
 Smith, Graham (June 15, 2019). "Wot I Think: Katana Zero". Rock, Paper, Shotgun. Archived from the original on February 8, 2022. Retrieved March 15, 2022.
 Saltzman, Mitchell (April 18, 2019). "Katana Zero Review". IGN. Archived from the original on April 18, 2019. Retrieved February 13, 2022.
 Vogel, Mitch (April 18, 2019). "Katana Zero Review (Switch eShop)". Nintendo Life. Archived from the original on April 19, 2019. Retrieved March 15, 2019.
 Morton, Lauren (April 30, 2019). "Katana Zero review". PC Gamer. Archived from the original on February 8, 2022. Retrieved March 15, 2022.
 Harris, Iain (April 18, 2019). "Katana Zero blends Gunpoint noir with Hotline Miami ultraviolence". PCGamesN. Archived from the original on September 30, 2021. Retrieved September 30, 2021.
 Andriessen, CJ (May 5, 2019). "Review: Katana Zero". Destructoid. Archived from the original on February 8, 2022. Retrieved March 15, 2022.
 Wallace, Chris (January 16, 2020). ""Artists are super hard to find as an indie developer" – Katana Zero's Justin Stander on his game's protracted development". MCV/Develop. Archived from the original on August 9, 2021. Retrieved October 2, 2021.
 Makedonski, Brett (January 18, 2019). "Katana Zero wears its Hotline Miami influences on its kimono". Destructoid. Archived from the original on September 30, 2021. Retrieved September 30, 2021.
 Doolan, Liam (May 18, 2019). "Katana Zero Is Receiving A "Significant Update" At The End Of This Month". Nintendo Life. Archived from the original on February 12, 2022. Retrieved February 12, 2022.
 HG101 2020, 20:00–30:00.
 Donlan, Christian (April 30, 2019). "Katana Zero review: It's Groundhog Day with a sword". Eurogamer. Archived from the original on September 23, 2022. Retrieved October 16, 2022.
 Grayson, Nathan (April 30, 2019). "My Favorite Katana Zero Character Is This Hapless Goon I Killed Near The Start Of The Game". Kotaku. Archived from the original on November 3, 2021. Retrieved October 16, 2022.
 Wright, Steven T. (August 5, 2019). "How Katana Zero cuts its way to a surprisingly deep story". Red Bull. Archived from the original on October 4, 2021. Retrieved September 28, 2021.
 Askiisoft (April 18, 2019). Katana Zero (Amazon Luna, macOS, Nintendo Switch, Windows, Xbox One). Devolver Digital. Level/area: Bunker Pt. 1.
 Watts, Steve (March 29, 2019). "PAX East 2019: Katana Zero Is A Stylish, Surprising Action Thriller". GameSpot. Archived from the original on January 20, 2021. Retrieved October 16, 2022.
 Hughes, William (May 3, 2019). "Katana Zero is a good game about swords, and a great game about being a rude, interrupting prick". The A.V. Club. Archived from the original on January 19, 2022. Retrieved October 16, 2022.
 Cunningham, James (January 21, 2019). "Building a World and Killing a Path Through it in Katana Zero". Hardcore Gamer. Archived from the original on February 11, 2022. Retrieved October 16, 2022.
 Askiisoft (April 18, 2019). Katana Zero (Amazon Luna, macOS, Nintendo Switch, Windows, Xbox One). Devolver Digital. Level/area: Mansion.
 Askiisoft (April 18, 2019). Katana Zero (Amazon Luna, macOS, Nintendo Switch, Windows, Xbox One). Devolver Digital. Level/area: Slaughterhouse.
 Williams, Mike (April 26, 2019). "Katana Zero Review". USgamer. Archived from the original on April 19, 2019. Retrieved March 15, 2022.
 Askiisoft (April 18, 2019). Katana Zero (Amazon Luna, macOS, Nintendo Switch, Windows, Xbox One). Devolver Digital. Level/area: Bunker Pt. 2.
 Askiisoft (April 18, 2019). Katana Zero (Amazon Luna, macOS, Nintendo Switch, Windows, Xbox One). Devolver Digital. Level/area: Credits roll.
 Leijon, Eric (February 2020). "Building a video game smash between classes". McGill Alumni. Archived from the original on April 9, 2022. Retrieved October 8, 2021.
 "YoYo Games launches major GameMaker Studio 2 update allowing for 'a more visual approach'". MCV/Develop. August 20, 2020. Archived from the original on August 7, 2022. Retrieved August 7, 2022.
 EdN (April 26, 2019). "askiisoft On Katana ZERO". PS4Blog.net. Archived from the original on September 27, 2021. Retrieved October 10, 2021.
 Sinclair, Brendan (May 27, 2020). "How does a hit like Katana Zero change things for a creator?". GamesIndustry.biz. Archived from the original on February 14, 2022. Retrieved February 12, 2022.
 Vincent, Brittany (March 21, 2019). "Katana Zero interview: One-hit kills make you feel "badass"". Shacknews. Archived from the original on September 27, 2021. Retrieved October 10, 2021.
 LeClair, Kyle (September 13, 2016). "PAX: Katana Zero May Be The True Hotline Miami Successor". Hardcore Gamer. Archived from the original on February 12, 2022. Retrieved February 12, 2022.
 Leri, Michael (April 18, 2019). "Katana Zero developer describes how he made a tight, punishing action game". GameRevolution. Archived from the original on September 27, 2021. Retrieved October 6, 2021.
 Ruhland, Perry (March 3, 2017). "Indie Interview - Katana ZERO". TechRaptor. Archived from the original on September 27, 2021. Retrieved October 10, 2021.
 Lemne, Bengt; Stander, Justin (March 16, 2016). Katana Zero - Justin Stander Interview - GDC 2016. Gamereactor. Archived from the original on February 15, 2022. Retrieved February 12, 2022 – via YouTube.
 Wiltshire, Alex (December 4, 2019). "How Katana Zero brought action into cutscenes". Rock, Paper, Shotgun. Archived from the original on September 30, 2021. Retrieved October 4, 2021.
 Brady, Sean (May 13, 2019). "On the "Katana ZERO" Score, Synthwave Goes Neo-Noir". Bandcamp. Archived from the original on October 24, 2021. Retrieved February 10, 2022.
 Barnes 2019, 0:00–10:00.
 Barnes 2019, 20:00–30:00.
 Barnes 2019, 40:00–47:13.
 Snow, Jeremy (May 28, 2019). "The Geekly Grind Interviews: Bill Kiley". The Geekly Grind. Archived from the original on October 4, 2021. Retrieved February 11, 2022.
 Barnes 2019, 0:00–10:00; 40:00–47:13.
 Barnes 2019, 10:00–20:00.

Main menu
 
WikipediaThe Free Encyclopedia

Search
Create account
Log in

Personal tools
Contents  hide
(Top)
Electoral system
Toggle Electoral system subsection
Voters
Contesting parties
Presidential election
Toggle Presidential election subsection
Legislative election
Toggle Legislative election subsection
Opinion polls
Toggle Opinion polls subsection
Finance and logistics
Toggle Finance and logistics subsection
Preliminary results
Official results
Toggle Official results subsection
Aftermath
Toggle Aftermath subsection
Reactions
Toggle Reactions subsection
Notes
References
External links
2024 Indonesian general election

7 languages
Article
Talk
Read
Edit
View history

Tools
From Wikipedia, the free encyclopedia
2024 Indonesian general election

← 2019	14 February 2024	2029 → 
Registered	204,421,612 (Increase 6.04%)
Turnout	82.39% (Increase 0.42pp)
Presidential election
 			
Candidate	Prabowo Subianto	Anies Baswedan	Ganjar Pranowo
Party	Gerindra	Independent	PDI-P
Alliance	Advanced Indonesia[a]	Change[b]	Alliance of Parties[c]
Running mate	Gibran Rakabuming Raka	Muhaimin Iskandar	Mahfud MD
Popular vote	96,214,691	40,971,906	27,040,878
Percentage	58.59%	24.95%	16.47%

Results by city/regency
President before election
Joko Widodo
PDI-P
Elected President
Prabowo Subianto
Gerindra
Legislative election
All 580 seats in the House of Representatives
291 seats needed for a majority
Party	Leader	%	Seats	+/–
PDI-P	Megawati Sukarnoputri	16.72	110	−18
Golkar	Airlangga Hartarto	15.29	102	+17
Gerindra	Prabowo Subianto	13.22	86	+8
PKB	Muhaimin Iskandar	10.62	68	+10
NasDem	Surya Paloh	9.66	69	+10
PKS	Ahmad Syaikhu	8.42	53	+3
Demokrat	Agus Harimurti Yudhoyono	7.43	44	−10
PAN	Zulkifli Hasan	7.24	48	+4
This lists parties that won seats. See the complete results below.

Results by electoral district
Speaker before	Speaker after
Puan Maharani
PDI-P	TBD
PDI-P
This article is part of a series on the
Politics of
Indonesia

National government
Legislature
Executive
Judiciary
Elections
Political parties
Foreign relations
flag Indonesia portalicon Politics portal
vte
General elections were held in Indonesia on 14 February 2024 to elect the president, vice president, People's Consultative Assembly (MPR) which consists of the House of Representatives (DPR), the Regional Representative Council (DPD), and members of local legislative bodies (DPRD) at the provincial and city/regency levels.[1][2] The newly elected members of the MPR will be sworn in on 1 October 2024, while the elected president and vice president will be sworn in on 20 October 2024.[3] Incumbent President Joko Widodo was ineligible to run for a third term due to limitations established by the Indonesian constitution.[4] The election has over 200 million eligible voters, voting in over 800,000 polling stations across the country on the same date.
Defense minister and retired army general Prabowo Subianto received a majority of the vote in the first round defeating his two rivals Anies Baswedan and Ganjar Pranowo. Prabowo's 96.2 million votes were the highest received by any candidate in a democratic election in Indonesia, surpassing Joko Widodo's 85.6 million votes won in the 2019 election.
Electoral system
The election was held in accordance with the Law No. 7 of 2017. The General Elections Commission (KPU), an independent statutory body was responsible for organizing the election.

Ballot papers for the election in South Tangerang, Banten
All voters were given five ballot papers: one for president and vice president, one for the House of Representatives (DPR), one for the Regional Representative Council (DPD), one for the Provincial Regional House of Representatives (DPRD Provinsi), and one for the City/Regency Regional House of Representatives (DPRD Kota/Kabupaten).[5] Voters in Jakarta received just four ballot papers,[6] while overseas voters received just two.[7] Voters used a nail to poke a hole in the ballot paper indicating which party or candidate they wished to vote for, and then dipped their fingers in ink as a precaution against voter fraud.[5]
Presidential
In order to run as a presidential candidate, a candidate had to be formally endorsed by a political party or a coalition thereof holding a minimum of 20 percent of seats in the DPR or having won at least 25 percent of the popular vote in the previous election, i.e. in the 2019 election.[8]
The voting procedure followed a two-round system, with voters simply choosing one of the candidate pairs. A winning candidate required a majority and at least 20% of the votes in over half of Indonesia's provinces to be declared the winner. If no candidate pairs had fulfilled the criterion (50%+1 of total popular votes), the election would have had to progress to a second round with only the two candidates receiving the most popular votes, which would have been held on 26 June.[5]
According to the Indonesian electoral law of 2017 and by the decision of the Constitutional Court of Indonesia number 90/PUU-XII/2023, presidential candidates have to:[9]
Be at least 40 years old; or have/are currently holding positions that elected through general elections including regional head elections[10]
Have been resident in Indonesia for at least 5 years; and
Not have held foreign citizenship, either at the time of the election or at any time before.
Legislative
Members of both the House of Representatives (DPR) and the Regional Houses of Representatives (DPRD) were elected from multi-member electoral districts through voting with an open list system, and seat distribution is done with the Sainte-Laguë method. There was a gender quota requiring at least 30% of registered candidates to be female.[11]
A 4% parliamentary threshold is set for parties to be represented in the DPR, though candidates could still win seats in the regional councils provided they won sufficient votes. There were 580 DPR seats contested. Nationally, there are 84 national electoral districts, with 301 provincial and 2,375 municipal electoral districts. Senatorial candidates for the DPD were not allowed to be members of any political party. Four senators were elected for each province – a total of 152 members from all 38 provinces.[12]
These were the first elections for provincial deputies and representatives of both Houses for Central Papua, Southwest Papua, South Papua, and Highland Papua - all new provinces formed in 2022. On 12 December 2022, Government Regulation in Lieu of Law No. 1/2022 signed and published to amend the 2017 electoral law to make the new electoral regions to those provinces and facilitate the election there.[13]
Nusantara, the designated new national capital, was not a new separate electoral region in the 2024 general elections as it is still under construction and therefore had an insufficient population for it to have its own electoral district. Therefore, the government decided that the DPR will serve as a temporary representation body until 2029, when Nusantara can be established as new electoral region. For the 2024 election, electors living within Nusantara were included in the East Kalimantan electoral region.[14][15][16]
Voters

A polling station in North Jakarta on election day
The voting age is 17, or less if the voter has an Indonesian biometric identity card or e-KTP through marriage.[17][18] However, since the age of marriage was amended to age 19 in 2019, there are no longer any married people under the age of 17.[19] Members of the Indonesian National Armed Forces (TNI) and the Indonesian Police (Polri) are not allowed to vote.[20] Around 33 percent of voters were Millennials, and 23 percent were part of Generation Z.[21]
On 18 April 2023, the KPU announced that there were provisionally 205,853,818 registered voters, including 1,574,737 voters registered overseas. It was planned that the vote would be held in 823,287 polling stations (TPS).[22] This was updated to a "final" figure of 204,807,222 voters in July 2023, who were to vote in 823,220 polling stations.[23]
Postal ballots were sent to Indonesian embassies overseas in early January 2024.[24] Although overseas voters cast their votes before voters in Indonesia, the KPU explicitly banned any exit polls or publication of results from overseas voting before the election process had been completed across Indonesia.[25]
Voting occurred between 7:00 and 13:00 local time, although voters who had arrived before 13:00 and were still in the queue were allowed to cast their votes after the deadline.[26]
Contesting parties
See also: List of political parties in Indonesia
To participate in the election, political parties had to have branches in every province in Indonesia, 75% of regencies or cities in those provinces, and 50% of districts in regencies where the party have branches.[27] In April 2022, the Ministry of Law and Human Rights declared the names of 75 national political parties eligible to register for the 2024 elections.[28][29] In the end, a total of 24 political parties registered with the KPU to run in the election nationally.[30] On 14 December 2022, the KPU announced that 17 parties would be eligible to contest the legislative election.
The Ummah Party, who the KPU deemed not qualified to participate in the elections, accused the KPU of irregularities in the process. The party subsequently filed a written complaint.[31] Following mediations brokered by Bawaslu between the party and the KPU on 20 and 21 December, Bawaslu instructed the electoral commission to repeat the verification process for Ummah Party.[32] The party declared as qualified to participate in the election on 30 December.[33][34]
Meanwhile, the Just and Prosperous People's Party (PRIMA), which registration was initially rejected, filed a lawsuit against KPU, and won the right for a second verification from the KPU.[35] However, on 19 April 2023, the KPU deemed PRIMA not qualified to participate in 2024 elections after the party failed in its factual verification phase, where the KPU found the party's membership numbers below the required threshold.[36] The Indonesian Justice and Unity Party and Berkarya Party also failed to qualify for the election, despite participating in 2019 and having had party members elected as members of regional legislatures then.[37][38]
#	English name
Indonesian name	Leader	2019 result
Votes (%)[citation needed]	Seats
1
PKB	National Awakening Party
Partai Kebangkitan Bangsa	Muhaimin Iskandar	9.69%	
58 / 575
2
Gerindra	Great Indonesia Movement Party
Partai Gerakan Indonesia Raya	Prabowo Subianto	12.57%	
78 / 575
3
PDI-P	Indonesian Democratic Party of Struggle
Partai Demokrasi Indonesia Perjuangan	Megawati Sukarnoputri	19.33%	
128 / 575
4
Golkar	Party of Functional Groups
Partai Golongan Karya	Airlangga Hartarto	12.31%	
85 / 575
5
NasDem	National Democrats Party
Partai Nasional Demokrat	Surya Paloh	9.05%	
59 / 575
6
PB	Labour Party
Partai Buruh	Said Iqbal	New
7
Gelora	Indonesian People's Wave Party
Partai Gelombang Rakyat Indonesia	Anis Matta	New
8
PKS	Prosperous Justice Party
Partai Keadilan Sejahtera	Ahmad Syaikhu	8.21%	
50 / 575
9
PKN	Nusantara Awakening Party
Partai Kebangkitan Nusantara	Anas Urbaningrum	New
10
Hanura	People's Conscience Party
Partai Hati Nurani Rakyat	Oesman Sapta Odang	1.54%	
0 / 575
11
Garuda	Change Indonesia Guardian Party
Partai Garda Perubahan Indonesia	Ahmad Ridha Sabana	0.50%	
0 / 575
12
PAN	National Mandate Party
Partai Amanat Nasional	Zulkifli Hasan	6.84%	
44 / 575
13
PBB	Crescent Star Party
Partai Bulan Bintang	Yusril Ihza Mahendra	0.79%	
0 / 575
14
Demokrat	Democratic Party
Partai Demokrat	Agus Harimurti Yudhoyono	7.77%	
54 / 575
15
PSI	Indonesian Solidarity Party
Partai Solidaritas Indonesia	Kaesang Pangarep	1.89%	
0 / 575
16
Perindo	Indonesian Unity Party
Partai Persatuan Indonesia	Hary Tanoesoedibjo	2.67%	
0 / 575
17
PPP	United Development Party
Partai Persatuan Pembangunan	Muhamad Mardiono	4.52%	
19 / 575
Ballot number 18-23 allocated to local parties in Aceh[39]
18
PNA	Aceh State Party
Partai Nanggroe Aceh	Irwandi Yusuf	DNP
19
Gabthat	Aceh's Generation Unite in Obedience and Piety Party
Partai Generasi Atjeh Beusaboh Tha'at dan Taqwa	Ahmad Tajuddin
20
PDA	Aceh Abode Party
Partai Darul Aceh	Muhibbussabri A. Wahab
21
PA	Aceh Party
Partai Aceh	Muzakir Manaf
22
PAS Aceh	Aceh Just and Prosperous Party
Partai Adil Sejahtera Aceh	Tu Bulqaini Tanjongan
23
SIRA	Independent Solidity of Acehnese Party
Partai Soliditas Independen Rakyat Aceh	Muslim Syamsuddin
24
Ummat	Ummah Party
Partai Ummat	Ridho Rahmadi	New
Presidential election
Main article: 2024 Indonesian presidential election
Candidates
In July 2017, the House of Representatives passed a law that only parties or coalitions with at least 20% of seats in the legislature (i.e. 115 seats), or 25% of votes in the previous election are eligible to submit a presidential candidate. Requirements for presidential/vice-presidential candidates are, Indonesian-born citizens, Indonesian citizens who were born abroad, a minimum age of 40 and a requirement to "have a belief in the One and Only God". If the candidates had spouses, they also had to be Indonesian citizens. A criminal record resulting in over five years of incarceration or an active bankruptcy bars a candidate from running.[40]
The Anies Baswedan–Muhaimin Iskandar and Ganjar Pranowo–Mahfud MD pairs officially registered with the General Elections Commission on 19 October 2023.[41] The Prabowo Subianto–Gibran Rakabuming pair officially registered on 25 October 2023.[42]
Nominees
01
2024 Coalition of Change ticket
Anies Baswedan	Muhaimin Iskandar
for President	for Vice President


Governor of Jakarta (2017–2022)	Deputy Speaker of the House of Representatives (2019–present)
Campaign

02
2024 Advanced Indonesia Coalition ticket
Prabowo Subianto	Gibran Rakabuming
for President	for Vice President


Minister of Defense (2019–present)
2014, 2019 presidential nominee	Mayor of Surakarta (2021–present)
Campaign

03
2024 Alliance of Parties ticket
Ganjar Pranowo	Mahfud MD
for President	for Vice President


Governor of Central Java (2013–2023)	Coordinating Minister for Political, Legal and Security Affairs (2019–2024)
Campaign

Withdrawn support
The National Awakening Party had previously declared support for Prabowo Subianto but later rescinded their support and declared support for Anies Baswedan with the National Awakening Party's Chairman, Muhaimin Iskandar, being selected as Anies Baswedan's running mate.[43][44]
Demokrat had previously declared support for Anies Baswedan, but due to the selection of Muhaimin Iskandar as Anies Baswedan's running mate, Demokrat Party's Chairman Agus Harimurti Yudhoyono rescinded their support and then declared support for Prabowo Subianto.[45][46]
The Indonesian Solidarity Party had previously declared their support for Ganjar Pranowo but rescinded support and on 24 October 2023, officially declared support for Prabowo Subianto[47][48]
Gibran's candidacy
An October 2023 ruling by the Constitutional Court of Indonesia added an exception to the 40-year minimum age criteria, allowing those younger than 40 who had been previously elected as regional leaders to run as presidential or vice-presidential candidates. This allowed 36-year-old Gibran Rakabuming, son of incumbent president Jokowi and mayor of Surakarta, to run for the vice-presidency. The ruling was controversial as the court chief justice, Anwar Usman, is Gibran's uncle.[49][50][51] Anwar Usman was ultimately demoted by the Majelis Kehormatan Mahkamah Konstitusi or the Honorary Council of the Constitutional Court from the position of Chief Justice on 8 November after finding him guilty of conflict of interest on the ruling.[52] Furthermore, the KPU was found to have committed ethics violations surrounding Gibran's vice presidential registration for allowing him to register his candidacy before the commission had adjusted the age minimum for candidates in its internal regulation.[53] A lawsuit was filed by the Indonesian Democracy Defenders (TPDI) and the Indonesian Advocates Movement (Perekat Nusantara) against Joko Widodo, Gibran Rakabuming, Anwar Usman and First Lady Iriana alleging nepotism and political dynasty on the part of the respondents, but was dismissed by the Jakarta State Administrative Court a day before the election.[54]
Debates
Five concurrent televised presidential and vice presidential debates were held between 12 December 2023 and 4 February 2024. During the debate on 21 January, Gibran Rakabuming was seen making a "ducking" gesture and pretending to search for a lost item in response to an answer from Mahfud MD, which drew mostly negative reactions online for its supposed rudeness.[55][56]
Social media usage and disinformation
Parts of this article are copied directly from 2024 Indonesian presidential election
Prabowo Subianto's campaign was noted for its efforts at rehabilitating his image from his association with human rights violations during the dictatorship of former President Suharto into a "gemoy" (cuddly) grandfather figure among the youth, going as far as to make an animated avatar of him on TikTok using artificial intelligence. Anies Baswedan's and Ganjar Pranowo's campaign also used interactive AI chatbots to engage with voters.[57][58][59]
During the campaign, Anies Baswedan was targeted by a deepfake audio recording purportedly showing him being chastised by a political backer in January. Prabowo Subianto's campaign team used AI to depict children in a television commercial in order to bypass laws prohibiting the appearance of minors in electoral advertisements.[57]
Golkar, one of the parties supporting Prabowo for president, uploaded a viral AI-generated deepfake video on social media of a simulation of Suharto, who had died in 2008, in which he appeared to urge voters to select the party's candidates in the upcoming election. This led some civil society organizations to urge the KPU to implement regulations on the usage of artificial intelligence.[60]
Allegations of state support
On 12 February 2024, investigative journalist Dandhy Laksono released a documentary on YouTube directed by him, titled Dirty Vote, alleging that Joko Widodo used state funds to support Prabowo Subianto's campaign, becoming viral within the day and prompting accusations of sabotage by Prabowo's campaign team.[61] The presidential office denied the claims, while protests were held in reaction to the allegations.[62]
Legislative election
Contested seats
Legislative elections in Indonesia: February 2024[63]
Level	Institution	Seats contested	Change from 2019 elections	Candidates running
National
Nasional	House of Representatives
Dewan Perwakilan Rakyat (DPR)	580	Increase5	9,917[64]
Regional Representative Council
Dewan Perwakilan Daerah (DPD)	152	Increase16[d]	668[64]
Provincial
Provinsi	Provincial People's Regional Representative Council
Dewan Perwakilan Rakyat Daerah Provinsi (DPRD I)	2,372	Increase165	32,880[66]
Regency/Municipal
Kabupaten/Kota	Regency/Municipal People's Regional Representative Council
Dewan Perwakilian Rakyat Daerah Kabupaten/Kota (DPRD II)	17,510	Increase170[e]	214,915[68]
Total	20,614	Increase356	258,380
Candidates
All legislative candidates has to be Indonesian citizens, over 21 years old, senior high school (or equivalent) graduates, and have never been convicted for a crime resulting in a sentence of five years or more. In addition, the candidates for the DPR or local legislatures has to be endorsed by a political party and are required to resign from their non-legislative government offices – except for the president and vice president – or their state-owned company positions. Legislators running for reelection or another body through a new political party are also required to resign.[69] For each electoral district, political parties are required to have at least 30 percent of running candidates, rounded to the closest whole number, be women. This was changed from the regulations in effect in the 2019 election, where the 30 percent figure would be rounded up, and thus less women candidates overall would be required.[70]
Candidate registration was opened between 1–14 May 2023, with a total of 10,341 candidates registering to run for the DPR. This included 17 of the 18 national parties registering a maximum of 580 candidates allowed each, with only the Gelora Party registering less with 481 candidates.[71] A total of 9,917 candidates were recognized by the KPU as DPR candidates.[72] Approximately 1,100 individuals registered as candidates for the Regional Representative Council, with only 622 passing requirements.[73]
Opinion polls
President
Main article: Opinion polling for the 2024 Indonesian presidential election
  Quick count   Real count
Pollster	Fieldwork date	Sample size	Margin of error			
Prabowo
Gerindra	Anies
Independent	Ganjar
PDI-P
14 February 2024	Election results	58.59%	24.95%	16.47%
Litbang Kompas[74]	14 February 2024			58.45%	25.25%	16.30%
Charta Politika[75]	14 February 2024			57.99%	25.36%	16.64%
SMRC[76]	14 February 2024	1,994		58.36%	24.86%	16.78%
Lembaga Survei Indonesia[77]	14 February 2024		1%	57.46%	25.30%	17.23%
Indikator[78]	14 February 2024	3,000	0.52%	58.17%	25.38%	16.46%
LSI Denny JA[79]	14 February 2024			58.47%	24.98%	16.55%
Poltracking[80]	14 February 2024	3,000	1%	58.51%	25.13%	16.36%
Populi Center[74]	14 February 2024		0.16%	59.08%	25.06%	15.86%
CSIS - Cyrus Network[81]	14 February 2024	2,000	1%	58.22%	24.94%	16.84%
Politika Research & Consulting[82]	14 February 2024			59.22%	24.07%	16.71%
SPIN[83]	5 - 8 February 2024	1,200	2.8%	54.8%	24.3%	16.1%
LSI Denny JA[84]	26 January - 6 February 2024	1,200	2.9%	53.5%	21.7%	19.2%
Lembaga Survei Indonesia[85]	29 January - 5 February 2024	1,220	2.9%	51.9%	23.3%	20.3%
4 February 2024	Fifth presidential debate
Indikator[86]	28 January - 4 February 2024	1,200	2.9%	51.8%	24.1%	19.6%
Populi Center[87]	27 January - 3 February 2024	1,500	2.53%	52.5%	22.1%	16.9%
Poltracking[88]	25 January - 2 February 2024	1,220	2.9%	50.9%	25.1%	18.4%
Lembaga Point Indonesia[89]	26 - 28 January 2024	1,500	2.53%	52.9%	22.7%	19.1%
Political Weather Station[90]	21 - 25 January 2024	1,220	2.81%	52.3%	21.3%	19.7%
LSI Denny JA[91]	16 - 26 January 2024	1,200	2.9%	50.7%	22%	19.7%
21 January 2024	Fourth presidential debate
Polling Institute[90]	15 - 16 January 2024	1,219	2.9%	48.7%	23%	20.9%
Indonesia Survey Center[92]	11 - 19 January 2024	1,670	2.4%	52%	21.7%	18.1%
Indikator[93]	10 - 16 January 2024	1,200	2.9%	48.6%	24.2%	21.6%
SPIN[94]	8 - 14 January 2024	2,178	2.1%	50.9%	18.7%	23.5%
Lembaga Survei Indonesia[93]	10 - 11 January 2024	1,206	2.9%	47.0%	23.2%	21.7%
Indonesia Polling Stations[95]	7 - 13 January 2024	1,220	2.8%	51.8%	21.3%	19.2%
Charta Politika[93]	4 - 11 January 2024	1,220	2.82%	42.2%	26.7%	28.0%
LSI Denny JA[96]	3 - 11 January 2024	1,200	2.9%	46.6%	22.8%	24.8%
7 January 2024	Third presidential debate
Indonesia Political Opinion[94]	1 - 7 January 2024	1,200	2.5%	42.3%	34.5%	21.5%
Poltracking[93]	1 - 7 January 2024	1,220	2.9%	46.7%	26.9%	20.6%
Indikator[96]	30 December 2023 - 6 January 2024	1,200	2%	45,8%	25,5%	23%
Ipsos Public Affairs[94]	27 December 2023 - 5 January 2024	2,000	2.19%	48.1%	21.8%	18.4%
Lembaga Survei Nasional[94]	28 December 2023 - 2 January 2024	1,420	2.6%	49.5%	24.3%	20.5%
Median[94]	23 December 2023 - 1 January 2024	1,500	2.53%	43.1%	26.8%	20.1%
Polling Institute[94]	26 - 28 December 2023	1,246	2.9%	46.2%	24.6%	21.3%
PRC[97]	20 - 27 December 2023	1,200	2.7%	42.4%	28.0%	21.8%
ICRC[94]	20 - 26 December 2023	1,230	2.79%	39.4%	25.6%	29.1%
Indikator[98]	23 - 24 December 2023	1,217	2.9%	46.7%	21.0%	24.5%
LSI Denny JA[98]	17 - 23 December 2023	1,200	2.9%	43.3%	25.3%	22.9%
22 December 2023	Second presidential debate
Polling Institute[98]	15 - 19 December 2023	2,130	2.9%	46.1%	22.1%	20.5%
CSIS[99]	13 - 18 December 2023	1,300	2.7%	43.7%	26.1%	19.4%
Puspoll[94]	11 - 18 December 2023	1,220	2.83%	41%	26.1%	27.6%
12 December 2023	First presidential debate
Indikator Publik[100]	3 - 11 December 2023	1,670	2.4%	50.2%	22.7%	23.1%
Poltracking[101]	29 November - 5 December 2023	1,220	2.9%	45.2%	23.1%	27.3%
Populi Center[102]	28 November - 5 December 2023	1,200	2.83%	46.7%	21.7%	21.7%
Litbang Kompas[103]	29 November - 4 December 2023	1,364	2.65%	39.3%	16.7%	15.3%
Indikator[104]	23 November - 1 December 2023	1,200	2.9%	38.2%	19.1%	20.4%
LSI Denny JA[105]	6 - 13 November 2023	1,200	2.90%	40.3%	20.3%	28.6%
Populi Center[106]	29 October - 5 November 2023	1,200	2.83%	43.1%	22.3%	23.0%
Poltracking[107]	28 October - 3 November 2023	1,220	2.9%	40.2%	24.4%	30.1%
Indikator[108]	27 October - 1 November 2023	1,220	2.9%	39.7%	24.4%	30.0%
Charta Politika[109]	26 - 31 October 2023	2,400	2.0%	34.7%	24.3%	36.8%
Indo Barometer[110]	25 - 31 October 2023	1,230	2.79%	43.5%	23.2%	33.3%
Legislature
Main article: Opinion polling for the 2024 Indonesian legislative election
This graph shows the polling trends in the run-up to the 2024 Indonesian legislative election. Scenario polls are not included.
The electoral threshold to obtain seats is currently set at 4%.

Finance and logistics

Workers unloading ballot boxes in Jakarta the day before the election.
The Indonesian Government budgeted Rp 25 trillion (~USD 1.7 billion) for the election preparations in 2022–2023, over half of which was used by the General Elections Commission (KPU) and most of the remaining funds used by the General Election Supervisory Agency.[111] The Ministry of Finance budgeted Rp 71.3 trillion for the whole election process, a 57 percent increase from the 2019 election's budget.[112] Around Rp 17 trillion (US$1.1 billion) of the budget is earmarked for presidential election runoffs, if one is required.[113]
Over 1.2 billion ballot papers were printed, along with 4.16 million ballot boxes.[114] According to KPU chairman Hasyim Asyari, the costs of printing the legislative ballots alone was over Rp 800 billion.[115] Ballots began to be printed in November 2023,[116] with the distribution of ballots mostly beginning on 10 February 2024. Polling stations are intended to receive their ballots the day before voting, although more isolated regions began distribution earlier.[117] To reach more isolated polling stations, helicopters, boats, and animal-drawn carts were employed.[118] KPU intends for each polling station to serve a maximum of 300 voters, although regulations allow for a maximum of 500. According to Hasyim Asyari, this was due to the time constraints at each polling station.[119] Each polling station has four voting booths.[120]
Over 5.7 million poll workers and volunteers (Kelompok Penyelenggara Pemungut Suara/KPPS) served at the polling stations in Indonesia and abroad. Due to concerns over deaths of KPPS staff in the 2019 election, KPU added rules in 2024 limiting their age to between 17 and 55, in addition to providing proof of good health.[121] Seven KPPS members are assigned to each polling station, with one serving as the head.[122] KPPS staff are paid Rp 1.1 million to 1.2 million (~USD 70) for their work, double the payment received by KPPS staff in 2019.[123] Vote counting at each polling station occurs between 14 and 15 February, with vote recapitulation being done between 15 February and 20 March at the village/subdistrict, district, and regency/city levels.[124] Each pair of presidential candidates are also allowed a maximum of two witnesses for each polling station.[125] The Indonesian National Police said that 4,992 personnel would be deployed to secure the counting of votes.[126]
As Indonesia's territory stretches across three time zones, voting began at 7:00 am in each time zone and closed at 01:00 pm., beginning at 22:00 GMT (13 February) in Papua and ending at 06:00 GMT (14 February) in Sumatra.[58]
Incidents
On 11 February, a mob in Paniai Regency, Central Papua, burned down a district office along with a number of ballots and ballot boxes over a KPU decision to relocate a polling station in the regency.[127] On 12 February, the KPU ordered the postponement of voting in 108 polling stations in Demak Regency, Central Java, due to flooding from the Wulan River.[128] On election day, voting was delayed by several hours in 34 polling stations in Jakarta due to flooding caused by a thunderstorm.[58] Voting was also postponed in some polling stations in South Tangerang due to flooding.[129][130] In total, 37,466 polling stations across the country began voting considerably after 07:00 am.[131] In Western New Guinea, polls were not held in 1,297 polling stations in Central Papua, Highland Papua and Papua Provinces due to problems related to logistics and social tensions revolving around the local noken system, in which a designated representative casts votes on behalf of a group.[132]
Voting was not held in one polling station in Cimahi as the ballot box delivered was found to be empty, while mixups of ballot papers were reported in other polling stations in the city.[133] In Bogor Regency, Bawaslu confirmed that eight ballot papers had been rigged to select certain candidates before they could be distributed to voters.[134] Bawaslu also confirmed that ballot tampering had occurred during overseas voting in Malaysia.[135] Migrant organizations in Malaysia also reported that ballots were being bought for between 25 and 50 ringgit (between US$5–10).[136] Bawaslu recorded around 1,200 electoral violations during the vote, mostly from ethical infractions and neutrality violations by government employees.[137]
Since 14 February, at least 57 election officers across the country have died from fatigue and work-related accidents and diseases during the counting of ballots.[138] Intimidation against election officers was reported in 1,473 polling stations, while 6,084 polling stations received mixed up ballots.[131]
During the vote-counting, allegations emerged of votes appearing larger on the KPU-generated online application Sirekap (Recapitulation Information System) than what the actual results showed. Bawaslu attributed the issue to possible errors on part of the newly founded app, and welcomed an audit into Sirekap.[139] The PDI-P announced its formal rejection of the use of Sirekap on 20 February.[140] Citing problems and discrepancies with Sirekap, the KPU ordered delays in the recapitulation of votes at the district level.[141]
Preliminary results
Official results are expected to be released in March, but quick counts from government-approved tabulators came out shortly after polling stations closed.[58] Initial tallies from Indikator Politik, Kompas, and the Lingkaran Survei Indonesia showed Prabowo Subianto receiving between 53.4 and 59.8 percent of votes cast, followed by Anies Baswedan, who received between 23.11 and 26.39 percent, and Ganjar Pranowo, who received between 16.72 and 17.12 percent.[142]
Official results
President
Candidate	Running mate	Party	Votes	%
Prabowo Subianto	Gibran Rakabuming Raka (Ind.)	Gerindra Party	96,214,691	58.59
Anies Baswedan	Muhaimin Iskandar (PKB)	Independent	40,971,906	24.95
Ganjar Pranowo	Mahfud MD (Ind.)	Indonesian Democratic Party of Struggle	27,040,878	16.47
Total	164,227,475	100.00
Valid votes	164,227,475	97.51
Invalid/blank votes	4,194,536	2.49
Total votes	168,422,011	100.00
Registered voters/turnout	204,421,612	82.39
Source: KPU
By province
Votes by province[143]	



Total votes
Anies Baswedan
Independent	Prabowo Subianto
Gerindra	Ganjar Pranowo
PDI-P
Votes	%	Votes	%	Votes	%
Sumatra	Aceh	2,369,534	73.56	787,024	24.43	64,677	2.01	3,221,235
North Sumatra	2,339,620	29.25	4,660,408	58.26	999,528	12.49	7,999,556
West Sumatra	1,744,042	56.53	1,217,314	39.45	124,044	4.02	3,085,400
Riau	1,400,093	37.96	1,931,113	52.35	357,298	9.69	3,688,504
Jambi	532,605	24.15	1,438,952	65.23	234,251	10.62	2,205,808
South Sumatra	997,299	18.98	3,649,651	69.47	606,681	11.55	5,253,631
Bengkulu	229,681	18.10	893,499	70.42	145,570	11.47	1,268,750
Lampung	791,892	15.49	3,554,310	69.55	764,486	14.96	5,110,688
Bangka Belitung Islands	204,348	23.08	529,883	59.85	151,109	17.07	885,340
Riau Islands	370,671	32.15	641,388	55.64	140,733	12.21	1,152,792
Java	Banten	2,451,383	34.02	4,035,052	55.99	720,275	9.99	7,206,710
Jakarta	2,653,762	41.07	2,692,011	41.67	1,115,138	17.26	6,460,911
West Java	9,099,674	31.68	16,805,854	58.50	2,820,995	9.82	28,726,523
Central Java	2,866,373	12.58	12,096,454	53.08	7,827,335	34.35	22,790,162
Yogyakarta	496,280	19.80	1,269,265	50.63	741,220	29.57	2,506,765
East Java	4,492,652	17.52	16,716,603	65.19	4,434,805	17.29	25,644,060
Kalimantan	West Kalimantan	718,641	22.34	1,964,183	61.05	534,450	16.61	3,217,274
Central Kalimantan	256,811	16.98	1,097,070	72.53	158,788	10.50	1,512,669
South Kalimantan	849,948	35.16	1,407,684	58.23	159,950	6.61	2,417,582
East Kalimantan	448,046	20.09	1,542,346	69.15	240,143	10.77	2,230,535
North Kalimantan	72,065	17.67	284,209	69.71	51,451	12.62	407,725
Lesser Sunda	Bali	99,233	3.70	1,454,640	54.26	1,127,134	42.04	2,681,007
West Nusa Tenggara	850,539	26.20	2,154,843	66.37	241,106	7.43	3,246,488
East Nusa Tenggara	153,446	5.27	1,798,753	61.80	958,505	32.93	2,910,704
Sulawesi	North Sulawesi	119,103	7.30	1,229,069	75.31	283,796	17.39	1,631,968
Gorontalo	227,354	29.39	504,662	65.24	41,508	5.37	773,524
Central Sulawesi	386,743	21.50	1,251,313	69.57	160,594	8.93	1,798,650
Southeast Sulawesi	361,585	23.09	1,113,344	71.11	90,727	5.79	1,565,656
West Sulawesi	223,153	27.23	533,757	65.14	62,514	7.63	819,424
South Sulawesi	2,003,081	37.94	3,010,726	57.02	265,948	5.04	5,279,755
Maluku	Maluku	228,557	21.16	665,371	61.59	186,395	17.25	1,080,323
North Maluku	200,459	26.85	454,943	60.93	91,293	12.23	746,695
Papua	Papua	67,592	10.81	378,908	60.62	178,534	28.56	625,034
West Papua	37,459	11.32	172,965	52.26	120,565	36.43	330,989
Southwest Papua	48,405	13.53	209,403	58.54	99,899	27.93	357,707
Central Papua	128,577	11.66	638,616	57.94	335,089	30.40	1,102,282
Highland Papua	284,184	21.89	838,382	64.56	175,956	13.55	1,306,740
South Papua	41,906	13.31	162,852	51.74	110,003	34.95	314,761
Overseas	125,110	18.64	427,871	63.73	118,385	17.63	671,366
Total	40,971,906	24.95	96,214,691	58.59	27,040,878	16.47	164,227,475
Demographics
The research and development department of Indonesian newspaper Kompas (Litbang Kompas) conducted an exit poll, and released a demographic breakdown based on political preference.
2024 Indonesian presidential election[144]
Social group	Anies
(%)	Prabowo
(%)	Ganjar
(%)	No answer
(%)	Lead
(%)
Gender
Male	21.7	53.6	15.7	9.0	31.9
Female	22.0	55.1	13.4	9.5	33.1
Age
17–25	16.7	65.9	9.6	7.8	49.2
26–33	20.2	59.6	11.7	8.5	39.4
34–41	22.3	54.1	13.9	9.7	31.8
42–55	24.3	49.1	14.0	12.0	24.8
56–74	25.7	43.1	21.3	9.9	17.4
Education
Primary	18.8	55.6	17.4	8.2	36.8
Secondary	20.7	57.4	12.3	9.6	36.8
Higher	34.3	41.7	12.6	11.4	7.4
Social class
Lower	19.7	55.9	16.0	8.4	36.2
Lower middle	21.0	55.9	14.4	8.7	34.9
Upper middle	25.3	50.9	11.3	12.5	25.6
Upper	30.4	45.6	15.1	8.9	15.2
Religion
Islam (Nahdlatul Ulama)	21.8	55.8	12.8	9.5	34.0
Islam (Muhammadiyah)	41.9	41.6	10.6	5.9	0.3
Islam (Others)	30.1	49.5	9.8	10.6	19.4
Catholic	1.7	64.9	29.3	4.1	35.6
Protestant	1.7	56.9	32.9	8.4	24.0
Hindu	0.0	47.5	43.2	9.4	4.3
Other	7.9	50.0	26.3	15.8	23.7
House of Representatives

Party	Votes	%	+/–	Seats	+/–
Indonesian Democratic Party of Struggle	25,387,279	16.72	–2.61	110	–18
Golkar	23,208,654	15.29	+2.98	102	+17
Gerindra Party	20,071,708	13.22	+0.65	86	+8
National Awakening Party	16,115,655	10.62	+0.93	68	+10
Nasdem Party	14,660,516	9.66	+0.61	69	+10
Prosperous Justice Party	12,781,353	8.42	+0.21	53	+3
Democratic Party	11,283,160	7.43	–0.34	44	–10
National Mandate Party	10,984,003	7.24	+0.40	48	+4
United Development Party	5,878,777	3.87	–0.65	0	–19
Indonesian Solidarity Party	4,260,169	2.81	+0.92	0	0
Perindo Party	1,955,154	1.29	–1.38	0	0
Gelora Party	1,281,991	0.84	New	0	New
People's Conscience Party	1,094,588	0.72	–0.82	0	0
Labour Party	972,910	0.64	New	0	New
Ummah Party	642,545	0.42	New	0	New
Crescent Star Party	484,486	0.32	–0.47	0	0
Garuda Party	406,883	0.27	–0.23	0	0
Nusantara Awakening Party	326,800	0.22	New	0	New
Total	151,796,631	100.00	–	580	+5
Source: KPU
By province

Seat results of the legislative elections by coalition in each province
38 provinces with a range of 3 to 91 seats in each
Province	Total
seats	Seats won
PDI-P	Golkar	Gerindra	Nasdem	PKB	PKS	PAN	Demokrat
Aceh	13	1	3	1	2	2	2	1	1
North Sumatra	30	6	8	4	3	2	2	2	3
West Sumatra	14	1	2	2	3	1	2	2	1
Riau	13	2	3	2	0	2	2	1	1
Jambi	8	1	2	1	1	1	0	1	1
South Sumatra	17	2	3	3	2	2	2	1	2
Bengkulu	4	1	1	0	1	0	0	1	0
Lampung	20	3	3	4	2	2	2	2	2
Bangka Belitung Islands	3	1	1	1	0	0	0	0	0
Riau Islands	4	1	1	1	1	0	0	0	0
Jakarta	21	4	2	3	1	2	5	3	1
West Java	91	11	17	16	8	13	12	8	6
Central Java	77	23	12	10	7	10	7	3	5
Yogyakarta	8	2	1	1	1	1	1	1	0
East Java	87	19	13	14	7	18	5	5	6
Banten	22	4	4	3	3	2	2	2	2
Bali	9	5	1	1	1	0	0	0	1
West Nusa Tenggara	11	1	1	1	2	2	2	1	1
East Nusa Tenggara	13	2	3	1	2	2	0	1	2
North Kalimantan	3	1	0	1	0	0	0	0	1
West Kalimantan	12	4	2	1	2	1	1	1	0
Central Kalimantan	6	1	1	1	1	0	0	1	1
South Kalimantan	11	0	3	2	2	0	1	3	0
East Kalimantan	8	1	2	1	1	1	1	1	0
North Sulawesi	6	2	1	1	1	0	0	0	1
Central Sulawesi	7	1	2	1	1	0	0	1	1
South Sulawesi	24	1	4	5	5	2	2	3	2
Southeast Sulawesi	6	1	1	1	1	1	0	0	1
Gorontalo	3	0	1	1	1	0	0	0	0
West Sulawesi	4	1	0	0	1	0	0	1	1
Maluku	4	1	0	1	0	0	1	1	0
North Maluku	3	1	1	0	0	0	1	0	0
Papua	3	1	0	1	1	0	0	0	0
West Papua	3	1	1	0	1	0	0	0	0
South Papua	3	1	0	0	1	1	0	0	0
Central Papua	3	1	1	0	1	0	0	0	0
Highland Papua	3	1	0	0	1	0	0	1	0
Soutwest Papua	3	0	1	0	1	0	0	0	1
Total	580	110	102	86	69	68	53	48	44
Regional Houses of Representatives
Main article: Results of the 2024 Indonesian Regional House of Representatives election
Provincial legislatures
Provincial legislature (DPRD Provinsi) election results
Municipal legislatures
Municipal legislature (DPRD Kabupaten/Kota) election results
Aftermath
Following the results of unofficial quick counts, Prabowo claimed victory on the evening of 14 February at an event with his supporters at Istora Senayan in Jakarta, calling it "the victory of all Indonesians."[183][184] Gibran Rakabuming also expressed thanks to Prabowo for "giving young people a chance."[62] Ganjar Pranowo's campaign team said that they were investigating reports of electoral violations and alleged "structural, systematic and massive fraud" during the voting.[185] Hasto Kristiyanto, the secretary-general of the PDI-P, said that election irregularities were enforced from the top down, beginning with the decision to allow Gibran Rakabuming Raka to run for vice-president. Hamdan Zoelva, former chief justice of the Constitutional Court and a member of Anies Baswedan's campaign team also said that there were "strong indications that violations occurred in a structured, systematic and massive way in the presidential election".[186] Remarks by independent observers indicated there were "no signs of systemic fraud".[187] Prabowo again expressed thanks to the electorate after the official confirmation of the election results on 20 March.[188]
The Indonesia Stock Exchange on 15 February recorded its sharpest rise in two months as quick count results indicated Prabowo's victory, which analysts attributed to the removal of political uncertainty which would arise from a runoff election. Largest gains were made by banks, nickel companies, and infrastructure firms.[189]
Protests
Following the allegations, police said that it would allow peaceful protests.[190] A demonstration was held in front of the Istana Merdeka in Jakarta in protest against Prabowo's claims of victory on 15 February, followed by rallies on 16 February against alleged electoral fraud[191] and Joko Widodo's perceived support for Prabowo at the KPU headquarters.[186] Joko Widodo dismissed the allegations of fraud, saying that evidence for fraud should be brought to Bawaslu and the Constitutional Court.[192] On the day the official election results were finally released on 20 March, 300 demonstrators protested alleged electoral fraud and Joko Widodo's support for Prabowo outside the KPU headquarters.[193]
Analysis
After the release of quick count results, Lingkaran Survei Indonesia attributed split-ticket voters to the Ganjar-Mahfud pair's poor performance despite the PDI-P's success in remaining the largest party in the legislature.[194] Notably, Prabowo won the most votes in the traditionally PDI-P supporting provinces of Central Java and Bali.[195] Kompas' exit polls found strong support for Prabowo's candidacy from non-Muslim voters and Nahdlatul Ulama Muslims, winning in 36 of 38 provinces (except for Aceh and West Sumatra, where the Anies-Muhaimin pair received the most votes).[196] Nahdlatul Ulama-affiliated academic Ulil Abshar Abdalla [id], in a Kompas column, attributed Prabowo's strong performance to Indonesian voters prioritizing the continuation of Jokowi's policies over concerns on legal and ethical violations.[197] Anies-Muhaimin and Ganjar-Mahfud were also defeated in East Java and Central Java, respectively, even though their supporting parties (PKB and PDI-P, respectively) won the most votes in the two provinces.[198] Prabowo also won the most votes in Bali, a traditional PDI-P stronghold, a victory attributed by analysts and Prabowo's campaign team to Jokowi's endorsement.[199][200]
Of parties which qualified for the House of Representatives in 2019, Golkar gained the most in 2024, increasing the party's vote share from 12 percent to over 15 percent.[201] Parties supporting Ganjar Pranowo – PDI-P. PPP, Hanura and Perindo – saw their vote shares decline from 2019, with PPP failing to qualify for parliament for the first time since the party's first electoral participation in the 1977 election.[202][203]
Reactions
Domestic
Outgoing president Joko Widodo stated that he had met and congratulated Prabowo and his own son, Gibran, on the evening of 14 February, based on quick count results.[204] Former president Susilo Bambang Yudhoyono congratulated Prabowo and stated that he "is now his commander".[205]
Chairman of the NasDem Party Surya Paloh in a press conference said his party accepted the results of both legislative and presidential elections and congratulated all winners of the legislative election and the Prabowo-Gibran ticket.[206] Despite this, Paloh states that NasDem will continue to support efforts to "seek justice" regarding the election results. NasDem will also file a lawsuit against the election results, including for the election of legislative members in six electoral districts, namely three electoral districts in Sumatra, one electoral district in Papua, and two electoral districts in Java.[207] The Prosperous Justice Party (PKS) also accepted the results with party secretary general Aboe Bakar Alhabsy expressing his happiness on its electoral gain of 3 seats.[208] However, PKS states that the legal process for the election is still ongoing, citing the problems of using the Sirekap.[208]
Anies Baswedan and Muhaimin Iskandar said that "It is important to safeguard the election process to ensure legitimacy, trust and inclusiveness in the results".[209] In an apparent criticism towards Gibran Rakabuming Raka's candidacy as vice president, Anies stated that "leaders born from a process tainted with fraud and irregularities will produce a regime that produces policies full of injustice" and his team did not want this to happen.[209] They have formally rejected the results of the presidential election announced by the KPU and intended to protest the result to the Constitutional Court.[210]
Ganjar Pranowo's campaign legal team deputy leader Todung Mulya Lubis also stated they would protest the results to the Constitutional Court and rejecting the results of the presidential election especially on the PDI-P's stronghold provinces of Central Java, Bali, North Sulawesi and East Nusa Tenggara.[211] Despite this, Todung Mulya Lubis stated they were not in the position to reject the whole results and only wanted to "correct the errors".[211] PDI-P's coalition partner PPP also rejected the election results, citing discrepancies between KPU's and the party's internal results.[212]
Grand Imam of the Istiqlal Mosque Nasaruddin Umar congratulated Prabowo and Gibran on their victory in the election and expressed hope that Indonesia will be more developed and more successful under their leadership.[213] Nahdlatul Ulama chief Yahya Cholil Staquf congratulated Prabowo-Gibran for winning the election and all parties that won seats in the legislative election.[214] Muhammadiyah chief Haedar Nashir had also congratulated Prabowo-Gibran, hoping that the elected pair have the spirit of a statesman in carrying out the popular mandate.[215]
Sultan and Governor of Yogyakarta Hamengkubuwono X congratulated Prabowo-Gibran for their electoral victory and expressed hope for their successful administration.[216]
International
Asia
 China – Chinese Ambassador to Indonesia Lu Kang visited Prabowo's home on 18 February and personally congratulated him over the election results while expressing hope that "Indonesia and China can grow together, prosper together".[217] On 20 March, President Xi Jinping delivered a congratulatory message to Prabowo for his victory and said that he looked forward to meet with him and work together with his administration.[218][219] This letter was delivered by Chinese Ambassador to Indonesia Lu Kang.[220]
 East Timor – President José Ramos-Horta congratulated Prabowo through a phone call on 19 February.[221]
 India – Prime Minister Narendra Modi congratulated the Indonesian people for the successful election and Prabowo for his victory on social media. Modi said he hoped to be able to work with the new president to strengthen the comprehensive strategic partnership between the two nation.[222]
 Iran – President Ebrahim Raisi delivered a congratulatory message to Prabowo for being elected president based on the election results. He also stated his hope for both countries to have collaborative efforts, mutual understanding, and shared endeavours under Prabowo's presidency.[223]
 Japan – Prime Minister Fumio Kishida delivered a congratulatory message to Prabowo and expressed his desire to encourage bilateral cooperation in handling regional and international situations.[224]
 Jordan – King Abdullah II congratulated Prabowo through a phone call based on the projected results. He also stated that Prabowo was needed by Indonesia and expressed his readiness to receive him in Jordan.[225]
 Malaysia – Prime Minister Anwar Ibrahim congratulated Prabowo for his victory in the election in a phone call and stated in his Twitter account that he was the first leader to congratulate him. He expressed belief that Prabowo can carry out the given mandate with excellence.[226][227]
 Palestine – President Mahmoud Abbas delivered a congratulatory message to Prabowo on winning the presidential election and expressed the commitment to working together towards further development and cooperation. He also stated Palestine's appreciation for Indonesia's steadfast support for the Palestinian cause and its people. The message was delivered by the Embassy of the Palestine in Jakarta.[228][229]
 Philippines – President Bongbong Marcos congratulated Prabowo for his commanding lead in the latest electoral count to be President on social media. He also stated that he looked forward to deepening bilateral ties, especially in the upcoming celebration of 75 years of diplomatic relations between Indonesia and the Philippines.[230][231][232]
 Saudi Arabia – King Salman of Saudi Arabia sent a cable of congratulations to Prabowo Subianto for winning the elections and wished him success.[233] Prince Mohammed bin Salman also sent a cable to Prabowo expressing his congratulations.[234]
 Singapore – President Tharman Shanmugaratnam congratulated Prabowo and stated that his strong mandate demonstrated the confidence and trust of the Indonesian people in his leadership.[235] Prime Minister Lee Hsien Loong congratulated Prabowo for his apparent victory, and congratulated Jokowi for the "smooth and successful conduct" of the election.[236] He also stated that he valued Prabowo's goodwill and friendship, and appreciated his insights.[235]
 South Korea – President Yoon Suk Yeol called Prabowo to congratulate him for winning the election. He also requested support for strengthening bilateral cooperation in various sectors and expressed hope for further efforts in spearheading freedom, peace and prosperity with Indonesia, considered one of South Korea's key partners in the Korea-ASEAN Solidarity Initiative (KASI).[237]
 Sri Lanka – President Ranil Wickremesinghe congratulated Prabowo for his victory in the election through a phone call.[226]
 Taiwan – President Tsai Ing-wen and Vice President Lai Ching-te congratulated Prabowo on his election victory through the Foreign Ministry. The ministry stated Indonesia and Taiwan shared the same democratic and liberal values and hoped to deepen bilateral ties with Indonesia under Prabowo.[238]
 Thailand – Prime Minister Srettha Thavisin congratulated Prabowo for his victory on social media. He expressed hope to strengthen bilateral relations.[239]
 Turkey – President Recep Tayyip Erdoğan delivered a congratulatory message to Prabowo following the election and expressed hope that the results will be auspicious. The message was delivered to Prabowo by Turkish Ambassador to Indonesia Talip Küçükcan.[240] Erdoğan also called Prabowo personally to congratulate him for winning the election.[241]
 United Arab Emirates – President Mohammed bin Zayed Al Nahyan congratulated Prabowo by phone call following the election results.[242]
Australasia & Oceania
 Australia – Foreign Minister Penny Wong said that the Australian government was looking "forward to working closely with the next president" when he is inaugurated.[59] Prime Minister Anthony Albanese called Prabowo on 15 February, tweeting that he was "the first foreign leader to speak today with Prabowo, who has a clear lead in official and unofficial counts".[243][244] On 23 February, Deputy Prime Minister and Minister of Defense Richard Marles personally congratulated Prabowo during his official visit to the Ministry of Defense of Indonesia in Jakarta.[245]
 New Zealand – Deputy Prime Minister and Foreign Minister Winston Peters congratulated Prabowo through his Twitter account and said he looked forward to strengthen comprehensive partnership between both nations.[246][247]
Europe
 Czech Republic – Prime Minister Petr Fiala congratulated Prabowo on social media and stated "readiness to strengthen bilateral relations".[248]
 France – President Emmanuel Macron congratulated Prabowo through a phone call and expressed his hope to celebrate 75 years of diplomatic relations between France and Indonesia.[249]
 Germany – Chancellor Olaf Scholz congratulated Prabowo for his victory and welcomed the opportunity to further developed the strategic partnership between Indonesia and Germany in peace and security, economic cooperation, and shared commitment against climate change.[250]
 Hungary – Prime Minister Viktor Orban congratulated Prabowo for his victory and hoped for better bilateral relations between both nations. The letter was delivered to Prabowo by Hungarian Ambassador to Indonesia Lilla Karsay.[251]
 Netherlands – Prime Minister Mark Rutte congratulated Prabowo following the projected outcome of the elections on social media. He also stated that he is looking forward to continuing to develop the friendship and strong bond between their countries.[252]
 Russia – President Vladimir Putin congratulated Prabowo on his election win and expressed confidence that Prabowo's administration would contribute to the further development of relations between their countries and strengthening security and stability in the Asia-Pacific region.[253]
 Serbia – President Aleksandar Vučić congratulated Prabowo through a phone call on 22 February.[254]
 Spain – Prime Minister Pedro Sánchez delivered a congratulatory letter to Prabowo following the election results and expressed confidence in Prabowo's experience and good performance to lead Indonesia in the future. The letter was delivered to Prabowo by Spanish Ambassador to Indonesia Francisco de Asis Aguilera Aranda.[255]
  Switzerland – President Viola Amherd delivered a congratulatory letter to Prabowo for his electoral victory as the eighth president of Indonesia. The letter was delivered by Swiss Ambassador to Indonesia Olivier Zehnder.[256]
 Ukraine – President Volodymyr Zelenskyy congratulated Prabowo by phone call following the election results and invited him to an upcoming summit on Ukraine's peace formula in Switzerland in April.[257]
 United Kingdom – Prime Minister Rishi Sunak delivered a congratulatory message to Prabowo on his electoral victory. The message was delivered to Prabowo's home by UK Ambassador to Indonesia Dominic Jermey.[258] Jermey also congratulated "hundreds of thousands of candidates who campaigned across the archipelago" and hailed the election as "a truly epic festival of democracy".[259]
Americas
 Nicaragua – President Daniel Ortega and Vice President Rosario Murillo congratulated Prabowo and said they hoped to strengthen ties between both countries.[260][261]
 United States – State Department spokesman Matthew Miller congratulated the Indonesian people "for their robust turnout" in the election, calling it "a testament to the durability and strength of the Indonesian people's commitment to the democratic process and electoral institutions".[262] When asked why the White House had yet to congratulate Prabowo on his victory, US National Security Council Communications Advisor John Kirby said a statement would be released at an appropriate time and will respect the will of the Indonesian people.[263] On 12 March, President Joe Biden delivered a congratulatory letter to Prabowo on his electoral victory which was delivered by US Ambassador to ASEAN Yohannes Abraham. He congratulated the Indonesian people for their successful election, calling it "a testament for commitment to democracy" and he looked forward to strengthening Indonesia–United States relations further.[264] On 22 March, Biden personally called Prabowo to congratulate him for winning the election.[265] Secretary of State Antony Blinken also congratulated Prabowo on his victory and said that he looked forward to partnering closely with the incoming government.[266][267]
Aside from Prabowo, Vice President-elect Gibran Rakabuming Raka also received words of congratulations from foreign government officials.[268]
Notes
 Includes Gerindra, the Golkar Party, the Democratic Party, the National Mandate Party, the Indonesian Solidarity Party, the Crescent Star Party, the Garuda Party, the Gelora Party, and the Aceh Party.
 Includes NasDem, the National Awakening Party, the Prosperous Justice Party, the Ummah Party, the Aceh Abode Party, the Independent Solidity of the Acehnese Party, and the Aceh Just and Prosperous Party.
 Includes PDI-P, the United Development Party, the Indonesian Unity Party, and the People's Conscience Party.
 Due to the formation of four new provinces in Western New Guinea: Central Papua, Highland Papua, South Papua, and Southwest Papua.[65]
 Due to population changes, 42 municipalities increased the size of their legislatures by 5, while 8 decreased theirs by 5.[67]
References
 Dewi, Retia Kartika (11 July 2022). "Jadwal Lengkap dan Tahapan Pemilu 2024". Kompas. Archived from the original on 19 February 2023. Retrieved 19 February 2023.
 Kiswondari (15 November 2020). "KPU Targetkan Sirekap Digunakan pada Pemilu 2024". sindonews.com (in Indonesian). Archived from the original on 17 January 2024. Retrieved 3 January 2021.
 "Indonesia Decides: 2024 Elections". The Jakarta Post. Archived from the original on 3 June 2023. Retrieved 14 June 2023.
 Wamad, Sudirman. "Jokowi soal 3 Periode: Saya Taat Konstitusi dan Kehendak Rakyat". detiknews (in Indonesian). Archived from the original on 18 September 2023. Retrieved 14 June 2023.
 "Indonesia election 2024: All you need to know". Al Jazeera. 9 February 2024. Archived from the original on 26 February 2024. Retrieved 21 March 2024.
 "Ingat! Surat Suara Pemilu 2024 untuk Warga DKI Jakarta Hanya 4, Ini Alasannya". KOMPAS.tv (in Indonesian). 13 February 2024. Archived from the original on 7 April 2024. Retrieved 7 April 2024.
 Wibawana, Widhia Arum (25 January 2024). "Pemilu 2024 di Luar Negeri Nyoblos Surat Suara Apa Saja? Ini Aturannya". detiknews (in Indonesian). Archived from the original on 17 March 2024. Retrieved 13 April 2024.
 "Apa yang perlu Anda ketahui tentang UU Pemilu". BBC News Indonesia (in Indonesian). 21 July 2017. Archived from the original on 27 March 2019. Retrieved 4 January 2021.
 Article 6 of the Constitution of Indonesia (1945)
 "Putusan MK Nomor 90/PUU-XXI/2023". Judgement No. 90/PUU-XXI of 2023 (PDF) (in Indonesian). Constitutional Court of Indonesia. Archived from the original (PDF) on 16 March 2024. Archived 16 March 2024 at the Wayback Machine
 Irham, Mast (8 February 2024). "Even with a 30% quota in place, Indonesian women face an uphill battle running for office". The Conversation. Archived from the original on 21 March 2024. Retrieved 21 March 2024.
 "PERATURAN PEMERINTAH PENGGANTI UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 1 TAHUN 2022 TENTANG PERUBAHAN ATAS UNDANG-UNDANG NOMOR 7 TAHUN 2017 TENTANG PEMILIHAN UMUM". Government Regulation in Lieu of Law No. 1 of 2022 (PDF) (in Indonesian). President of Indonesia. Archived 25 March 2023 at the Wayback Machine
 Rizqo, Kanavino Ahmad. "Jokowi Terbitkan Perppu Pemilu terkait 4 Daerah Otonomi Baru di Papua". detiknews (in Indonesian). Archived from the original on 13 December 2022. Retrieved 13 December 2022.
 Firmansyah, Muhammad Julnis (31 August 2022). Wibowo, Eko Ari (ed.). "Mendagri Usul Pemilu 2024 Tak Digelar Dulu di IKN, Ini Alasannya". Tempo (in Indonesian). Archived from the original on 18 November 2022. Retrieved 18 November 2022.
 Mantalean, Vitorio (31 August 2022). Prabowo, Dani (ed.). "IKN Tak Gelar Pemilu 2024, Mendagri Usul Badan Otorita Diawasi DPR RI". KOMPAS.com (in Indonesian). Archived from the original on 18 November 2022. Retrieved 18 November 2022.
 "Pakar: IKN Nusantara Tak Bisa Gelar Pemilu 2024". nasional (in Indonesian). Archived from the original on 18 November 2022. Retrieved 18 November 2022.
 deAcostaNoya, Rachel Wilson, Rosa (11 February 2024). "How Indonesia's future is in the hands of the young, in 5 charts". CNN. Archived from the original on 2 April 2024. Retrieved 18 April 2024.
 Asy'ari, Hakim; Syarifah, Nur (24 October 2022). "Peraturan Komisi Pemilihan Umum Nomor 7 Tahun 2022 tentang Penyusunan Daftar Pemilih dalam Penyelenggaraan Pemilihan Umum dan Sistem Informasi Data Pemilih" [General Election Commission Regulations Number 7 of 2022 about Preparation of Voter List In The Organization of Elections General and Voter Data Information System] (PDF). Jakarta: General Elections Commission. p. 4−11. Archived (PDF) from the original on 6 January 2024. Retrieved 18 November 2023.
 "Indonesia". Girls Not Brides. 26 August 2022. Archived from the original on 25 July 2023. Retrieved 3 March 2024.
 Melani, Agustina (12 February 2023). "Pemilu 2024, Apa Saja Syarat Pemilih? Simak di Sini". liputan6.com (in Indonesian). Archived from the original on 18 April 2023. Retrieved 18 April 2023.
 Setiawati, Susi (5 December 2023). "Gen Z-Milenial Wajib Bangga! Anda Jadi Penentu Next Presiden". CNBC Indonesia (in Indonesian). Archived from the original on 21 February 2024. Retrieved 21 February 2024.
 Simanjuntak, Surya Dua Artha (18 April 2023). Suwiknyo, Edi (ed.). "KPU: Daftar Pemilih Sementara Pemilu 2024 Capai 205 Juta". Bisnis.com (in Indonesian). Archived from the original on 18 April 2023. Retrieved 18 April 2023.
 "Jumlah Daftar Pemilih Tetap Pemilu 2024 di Seluruh Provinsi". CNN Indonesia (in Indonesian). 3 July 2023. Archived from the original on 15 February 2024. Retrieved 4 July 2023.
 Basyari, Iqbal (3 January 2024). "Surat Suara Dikirim, Bawaslu Diminta Awasi Pemilu di Luar Negeri". Kompas (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 "KPU Larang "Exit Poll" Luar Negeri Diumumkan Sebelum Pemilu WIB Selesai". KOMPAS.com (in Indonesian). 12 February 2024. Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Aisyah, Novia (13 February 2024). "Jadwal Buka-Tutup TPS Pemilu 2024 & Dokumen-Tata Cara Coblos, Cek di Sini!". detikedu (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Mantalean, Vitorio (18 August 2022). Saptohutomo, Aryo Putranto (ed.). "Syarat Partai Politik Ikuti Pemilu 2024". KOMPAS.com (in Indonesian). Archived from the original on 14 July 2023. Retrieved 14 July 2023.
 Savitri, Putu Indah (12 April 2022). Soebanto, Herry (ed.). "Ketua KPU: 75 parpol berhak mendaftar jadi peserta Pemilu 2024". Antara News (in Indonesian). Archived from the original on 25 April 2023. Retrieved 25 April 2023.
 "Deretan 75 Parpol yang Berhak Daftar Pemilu 2024". CNN Indonesia (in Indonesian). 24 April 2022. Archived from the original on 15 February 2024. Retrieved 25 April 2023.
 Mantalaen, Vitorio (15 August 2022). "40 Parpol Daftar Pemilu 2024, 24 Lanjut Verifikasi". detiknews (in Indonesian). Archived from the original on 15 December 2022. Retrieved 15 December 2022.
 Mantalean, Vitorio (14 December 2022). Asril, Sabrina (ed.). "Resmi, 17 Parpol Lolos Jadi Peserta Pemilu 2024, Ini Daftarnya..." Kompas. Archived from the original on 15 December 2022. Retrieved 15 December 2022.
 Ihsan, Nabil; Meilani A, Tri; Haryati, Sri (21 December 2022). Haryati, Sri (ed.). "Bawaslu instructs KPU to repeat verification process for Ummah Party". Antara News. Archived from the original on 21 December 2022. Retrieved 21 December 2022.
 Tandiah, Kenzu; Meilani A, Tri (30 December 2022). Haryati, Sri (ed.). "Ummah Party passes KPU re-verification to contest 2024 elections". Antara News. Archived from the original on 30 December 2022. Retrieved 30 December 2022.
 Silvia Ng (30 December 2022). "Lolos Peserta Pemilu 2024, Partai Ummat Pegang Nomor Urut 24". detiknews (in Indonesian). Archived from the original on 31 December 2022. Retrieved 30 December 2022.
 Wiryono, Singgih (1 April 2023). Santosa, Bagus (ed.). "KPU Nyatakan Prima Lolos Verifikasi Administrasi Peserta Pemilu 2024". KOMPAS.com (in Indonesian). Archived from the original on 4 April 2023. Retrieved 5 April 2023.
 Ameliya, Tri Meilani (19 April 2023). Noor, Chandra Hamdani (ed.). "KPU nyatakan Prima tak penuhi syarat untuk ikuti verfak perbaikan". Antara (in Indonesian). Archived from the original on 21 April 2023. Retrieved 21 April 2023.
 "Dilema Anggota DPRD dari Parpol Tak Lolos Verifikasi" (in Indonesian). Constitutional Court of Indonesia. 31 August 2023. Retrieved 18 February 2024.
 "Tidak Lolos Pemilu 2024, Partai Berkarya Buka Kemungkinan Melebur Jadi Ormas". Metro Jambi (in Indonesian). 14 March 2023. Archived from the original on 18 February 2024. Retrieved 18 February 2024.
 Arjanto, Dwi (2 January 2023). "Deretan 6 Partai Politik Lokal Aceh yang Lolos Pemilu 2024 dan Asal-usulnya". Tempo (in Indonesian). Archived from the original on 5 April 2023. Retrieved 5 April 2023.
 Wibawana, Widhia Arum (3 February 2023). "Syarat Calon Presiden 2024 Sesuai UU, Ini Isi Lengkapnya". detiknews (in Indonesian). Archived from the original on 30 October 2023. Retrieved 20 October 2023.
 "Pilpres 2024: Anies-Muhaimin dan Ganjar-Mahfud MD daftar capres-cawapres ke KPU". BBC News Indonesia (in Indonesian). 19 October 2023. Archived from the original on 31 October 2023. Retrieved 20 October 2023.
 Rahman, Faisal (25 October 2023). "Sah! Prabowo-Gibran Daftar Capres-Cawapres Pilpres 2024". CNBC Indonesia (in Indonesian). Archived from the original on 28 October 2023. Retrieved 28 October 2023.
 Azmi, Faiq (1 September 2023). "PKB soal Koalisi dengan Gerindra: Namanya Bukan KKIR, Otomatis Cabut". detiknews (in Indonesian). Archived from the original on 2 September 2023. Retrieved 2 September 2023.
 Mantalean, Vitorio (19 October 2023). Gonsaga, Aloysius (ed.). "Anies-Muhaimin Resmi Daftar Bakal Capres-Cawapres ke KPU RI". KOMPAS.com (in Indonesian). Archived from the original on 20 October 2023. Retrieved 25 October 2023.
 Dirgantara, Adhyasta (1 September 2023). Setuningsih, Novianti (ed.). "Demokrat Resmi Cabut Dukungan untuk Anies Baswedan". KOMPAS.com (in Indonesian). Archived from the original on 1 September 2023. Retrieved 1 September 2023.
 "Demokrat Pastikan Dukung Duet Prabowo-Gibran di Pilpres 2024". C. N. N. Indonesia (in Indonesian). Archived from the original on 15 February 2024. Retrieved 25 October 2023.
 Dirgantara, Adhyasta (22 August 2023). Setuningsih, Novianti (ed.). "PSI Resmi Batal Dukung Ganjar Capres, Akan Serap Ulang Aspirasi Rakyat". KOMPAS.com (in Indonesian). Archived from the original on 2 September 2023. Retrieved 2 September 2023.
 Guritno, Tatang (24 October 2023). Prabowo, Dani (ed.). "PSI Resmi Dukung Prabowo-Gibran di Pilpres 2024". KOMPAS.com (in Indonesian). Archived from the original on 2 November 2023. Retrieved 25 October 2023.
 "Indonesian court rules on presidential candidate eligibility". Al Jazeera. 16 October 2023. Archived from the original on 20 October 2023. Retrieved 20 October 2023.
 Janti, Nur (16 October 2023). "BREAKING: Constitutional Court opens door for Jokowi's son to run in presidential poll". The Jakarta Post. Archived from the original on 16 October 2023. Retrieved 22 October 2023.
 Tarigan, Edna; Karmini, Niniek (16 October 2023). "Indonesia's top court rules against lowering presidential candidates' age limit, but adds exception". Associated Press. Archived from the original on 23 October 2023. Retrieved 22 October 2023.
 "Top judge demoted in Indonesia after ruling favouring president's son". Al Jazeera. Archived from the original on 13 December 2023. Retrieved 12 December 2023.
 Suhenda, Dio; Janti, Nur (5 February 2024). "KPU found guilty of ethics breach in handling of Gibran VP bid". The Jakarta Post. Archived from the original on 5 February 2024. Retrieved 12 February 2024.
 "Court Dismisses Lawsuit against Jokowi over Political Dynasty, Nepotism". Tempo. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 Siregar, Kiki (5 February 2024). "Indonesia Elections 2024: High stakes as presidential candidates face off in last TV debate". CNA. Archived from the original on 5 February 2024. Retrieved 5 February 2024.
 "Dent in public hype over Indonesia VP candidate Gibran after 'rude' gesture against opponent in live debate". CNA. 22 January 2024. Archived from the original on 5 February 2024. Retrieved 5 February 2024.
 "AI 'resurrects' long dead dictator in murky new era of deepfake electioneering". CNN. 11 February 2024. Archived from the original on 12 February 2024. Retrieved 12 February 2024.
 "Indonesia votes for president with ex-general Subianto the favourite". RFI. 14 February 2024. Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 "Ex-general Prabowo Subianto poised for sweeping presidential win in Indonesia". France 24. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Fake Suharto video fuels debate on AI use in Indonesian election campaign". Benar News. 12 February 2024. Archived from the original on 12 February 2024. Retrieved 12 February 2024.
 "Indonesia Documentary Claims Widodo Improperly Backed Election Favourite". Barron's. Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 "Prabowo Subianto claims victory in Indonesia's election, as counting continues in world's largest single-day vote". ABC Australia. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Dapil dan Jumlah Kursi Anggota DPR dan DPRD dalam Pemilu Tahun 2024" (in Indonesian). General Elections Commission. 9 February 2023. Archived from the original on 18 April 2023. Retrieved 18 April 2023.
 Lingga, Rivan Awal (3 November 2023). Ninditya, Fransiska (ed.). "KPU tetapkan 9.917 DCT anggota DPR RI di Pemilu 2024". Antara News (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Muliawati, Anggi (28 December 2022). "Ada Pemekaran 4 Provinsi Papua, Kursi DPD di Pemilu 2024 Tambah 16". detiknews (in Indonesian). Archived from the original on 18 April 2023. Retrieved 18 April 2023.
 "Daftar Caleg DPRD Provinsi di Seluruh Indonesia". goodkind.id (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 "Kursi DPRD Se-Indonesia Bertambah Jadi 17.510". Republika (in Indonesian). 21 November 2022. Retrieved 15 April 2024.
 "Daftar Caleg DPRD Kabupaten/Kota di Seluruh Indonesia". goodkind.id (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Wahyuni, Willa (19 October 2022). "Minat Menjadi Caleg 2024? Begini Syaratnya Menurut Undang-Undang". hukumonline.com (in Indonesian). Archived from the original on 18 April 2023. Retrieved 18 April 2023.
 Saubani, Andri (7 May 2023). "Jumlah Anggota Dewan Perempuan Terancam Berkurang Akibat Peraturan Baru KPU". Republika (in Indonesian). Archived from the original on 18 May 2023. Retrieved 23 May 2023.
 Nurrahman, Aldiansyah (20 May 2023). Zagoto, Nofanolo (ed.). "Sebanyak 10.341 Orang Daftar Bakal Caleg DPR 2024". validnews.id (in Indonesian). Archived from the original on 22 May 2023. Retrieved 11 July 2023.
 Mulyanto, Randy (13 February 2024). "'Fix it from within': More Chinese Indonesians chase seats in parliament". Al Jazeera. Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Nababan, Willy Medi Christian (24 April 2023). "Jumlah Kandidat DPD di Pemilu 2024 Dipastikan Turun Lagi". Kompas (in Indonesian). Archived from the original on 14 July 2023. Retrieved 14 July 2023.
 "HITUNG CEPAT: Pemilihan Presiden dan Pemilihan Umum Legislatif dari Litbang KOMPAS dan Lembaga Survei Lainnya 2024" (in Indonesian).
 "Hasil Akhir Hitung Cepat Pemilihan Presiden 2024 Charta Politika Indonesia".
 "Trend Elektabilitas Capres dan Partai Politik Menjelang Pemilu 2024" (in Indonesian).
 "Hasil Akhir Quick Count Pilpres 2024 Versi LSI Seluruh Indonesia" (in Indonesian).
 "Quick Count Pemilu 2024" (PDF) (in Indonesian).
 "Quick Count LSI Denny JA 100%: Prabowo-Gibran 58,47%" (in Indonesian). 16 February 2024.
 "Quick Count Pilpres 2024 Poltracking Indonesia, dengan sampel 3.000 TPS" (in Indonesian).
 "Quick Count Pemilu 2024" (PDF) (in Indonesian).
 "Hasil Quick Count Pemilu 2024" (in Indonesian).
 "Survei SPIN Terbaru: Prabowo 54,8%, Anies 24,3%, Ganjar 16,1%". CNN Indonesia (in Indonesian). Retrieved 10 February 2024.
 "LSI Denny JA: Peluang Prabowo-Gibran Menang Satu Putaran Makin Terbuka". Detik (in Indonesian). Retrieved 10 February 2024.
 "Hasil Survei LSI Terbaru Jelang Pencoblosan Pilpres 2024, Siapa Pemenangnya?". Detik (in Indonesian). Retrieved 10 February 2024.
 "Survei Indikator: Elektabilitas Prabowo-Gibran 51,8 Persen, Anies-Muhaimin 24,1 Persen, Ganjar-Mahfud 19,6 Persen". Kompas (in Indonesian). Retrieved 10 February 2024.
 "6 Hasil Survei Terbaru Capres RI 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). Retrieved 8 February 2024.
 "Poltracking: Prabowo-Gibran 50,9%, AMIN 25,1%, Ganjar-Mahfud 18,4%". Detik (in Indonesian). Retrieved 10 February 2024.
 "Hasil Survei Terbaru Capres 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). Retrieved 1 February 2024.
 Dian, Rusti (29 January 2024). Amril, Rizal (ed.). "Survei Elektabilitas Capres dan Cawapres Terbaru 2024, Prabowo-Gibran Masih Tertinggi". Narasi Tv (in Indonesian). Retrieved 29 January 2024.
 Saptohutomo, Aryo Putranto (31 January 2024). "Survei LSI Denny JA: Anies-Muhaimin 22 Persen, Prabowo-Gibran 50,7 Persen, Ganjar-Mahfud 19,7 Persen". KOMPAS.com (in Indonesian). Retrieved 1 February 2024.
 "Hasil 17 Survei Terbaru Pilpres 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). Retrieved 29 January 2024.
 "8 Hasil Survei Terbaru Elektabilitas Anies, Prabowo, Ganjar Januari". CNN Indonesia (in Indonesian). Retrieved 22 January 2024.
 "Hasil 11 Survei Terbaru Pilpres 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). Retrieved 18 January 2024.
 "Hasil Survei Terbaru Capres 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). Retrieved 18 January 2024.
 "Hasil 13 Survei Terbaru Pilpres 2024: Anies Vs Prabowo Vs Ganjar". CNBC Indonesia (in Indonesian). 19 January 2024. Retrieved 19 January 2024.
 "Hasil Survei Politika Terbaru: Prabowo 42,4%, Ganjar dan Anies Saling Pepet". detik.news (in Indonesian). Retrieved 7 January 2024.
 "Elektabilitas Capres Anies, Prabowo, Ganjar di 4 Survei Terbaru". CNN Indonesia (in Indonesian). Retrieved 3 January 2024.
 Putri, Zunita (27 December 2023). "Survei CSIS Terbaru Elektabilitas Capres 2024 Pasca Debat, Ini Pemenangnya". detikbali (in Indonesian). Retrieved 28 December 2023.
 Anggrainy, Firda Cynthia, ed. (23 December 2023). "Survei Indikator Publik: Prabowo-Gibran Bisa Menang Satu Putaran!". Detik.com (in Indonesian). Retrieved 25 December 2023.
 Saptohutomo, Aryo Putranto, ed. (11 December 2023). "Survei Poltracking Indonesia Prediksi Pilpres Berlangsung 2 Putaran". KOMPAS.com (in Indonesian). Retrieved 22 December 2023.
 Permana, Rakhmad Hidayatulloh (11 December 2023). "Survei Populi Center: Prabowo-Gibran 46,7%, Ganjar-Mahfud 21,7%, AMIN 21,7%". detiknews (in Indonesian). Retrieved 22 December 2023.
 Guritno, Tatang (11 December 2023). "Survei Litbang "Kompas": Elektabilitas Prabowo-Gibran 39,3 Persen, Anies-Muhaimin 16,7 Persen, Ganjar-Mahfud 15,3 Persen". Kompas.com (in Indonesian). Retrieved 11 December 2023.
 Santika, Erlina Fury (11 December 2023). "Indikator Politik: Prabowo Jadi Top of Mind Capres 2024 | Databoks". katadata.co.id (in Indonesian). Retrieved 22 December 2023.
 Hutajulu, Matius Alfons (20 November 2023). "Survei LSI Denny JA: Prabowo-Gibran 40,3%, Ganjar-Mahfud 28,6%, AMIN 20,3%". Detik (in Indonesian). Retrieved 22 November 2023.
 Nufus, Wilda Hayatun (9 November 2023). "Survei Populi Center: Prabowo-Gibran Mungkin Menang 1 Putaran". Detik (in Indonesian). Retrieved 9 November 2023.
 Nufus, Wilda Hayatun (10 November 2023). "Survei Poltracking: Prabowo-Gibran 40,2%, Ganjar-Mahfud 30,1%, AMIN 24,4%". Detik (in Indonesian). Retrieved 10 November 2023.
 "Efek Gibran dan Dinamika Elektoral Terkini" (PDF). Indikator (in Indonesian). Retrieved 15 November 2023.
 "Peta Elektoral Pasca Putusan MK & Pendaftaran Capres - Cawapres" [The Electoral Map After The Constitutional Court Ruling] (PDF). Charta Politika (in Indonesian). Retrieved 6 November 2023.
 Luxiana, Kadek Melda (11 November 2023). "Survei Indo Barometer: Kemungkinan Pilpres 2024 1 Putaran". Detik (in Indonesian). Retrieved 11 November 2023.
 "Anggaran Pemilu Serentak 2024 Sebesar Rp 25 Triliun, Paling Besar KPU". liputan6.com (in Indonesian). 3 February 2023. Archived from the original on 11 July 2023. Retrieved 11 July 2023.
 "Melihat Anggaran Pemilu 2024 yang Sentuh Rp71 T, Untuk Apa Saja?". CNN Indonesia (in Indonesian). 12 February 2024. Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Raharjo, Agus (18 September 2023). "Biaya Pilpres 2024 Putaran Kedua Rp 17 Triliun". Republika (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Faisal, Ahmad Dani (21 September 2023). "Jumlah Surat Suara yang Dicetak KPU RI pada Pemilu 2024 0 : Foto Okezone Infografis". Okezone.com (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Satryo, Ahmad (27 February 2023). Tranggana, Angga Ulung (ed.). "Ketua KPU RI: Anggaran Cetak Surat Suara Pemilu dengan Sistem Terbuka Rp 803 Miliar". rmol.id (in Indonesian). Archived from the original on 11 July 2023. Retrieved 11 July 2023.
 Rahim, Annisa Aulia (30 October 2023). "KPU: Surat Suara Mulai Dicetak 10 November". detiknews (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 "KPU Targetkan Hari Ini Logistik Sampai ke TPS". Kompas (in Indonesian). 12 February 2024. Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Teresia, Ananda (13 February 2024). "On eve of Indonesia vote, defence minister ahead despite protests". Reuters. Retrieved 13 February 2024.
 "Pemilu 2024, KPU Rancang 1 TPS Maksimal 300 Pemilih". KOMPAS.com (in Indonesian). 3 October 2022. Archived from the original on 18 February 2024. Retrieved 18 February 2024.
 Wibawana, Widhia Arum (31 January 2024). "Berapa Jumlah Bilik Suara Per TPS di Pemilu 2024? Simak Infonya". detiknews (in Indonesian). Archived from the original on 18 February 2024. Retrieved 18 February 2024.
 Singgih, Viriya (12 February 2024). "Pemilu 2024: Bagaimana melindungi kesehatan petugas KPPS agar pemilu 2024 tak jadi 'kuburan massal' lagi". BBC News Indonesia (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Ibnu, Farhan (6 December 2023). Irawan, Iwan Bagus (ed.). "Ini Tugas Anggota KPPS 1 sampai 7 dalam Pemilu". Radio Republik Indonesia (in Indonesian). Retrieved 13 February 2024.
 Ibnu, Farhan (31 January 2024). Irawan, Iwan Bagus (ed.). "Rincian Gaji KPPS 2024 Terbaru dan Tugasnya". Radio Republik Indonesia (in Indonesian). Retrieved 13 February 2024.
 Wibawana, Widhia Arum (13 February 2024). "Tata Cara dan Jadwal Rekapitulasi Hasil Penghitungan Suara". detiknews (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Nababan, Willy Medi Christian (12 February 2024). "Sedikitnya Bakal Ada 4,8 Juta Saksi dari Timses di Seluruh TPS". Kompas (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 "Nearly 5,000 Police Officers will Secure Vote Count until Election Results Announced". Tempo. 19 March 2024. Archived from the original on 19 March 2024. Retrieved 19 March 2024.
 Suwandi, Dhias (13 February 2024). Hartik, Andi (ed.). "Massa Bakar Kotak dan Surat Suara Pemilu 2024 di Paniai". KOMPAS.com (in Indonesian). Archived from the original on 13 February 2024. Retrieved 13 February 2024.
 Nashr, Jamal Abdun (12 February 2024). "Terendam Banjir, 108 TPS di Demak Diusulkan Pemungutan Susulan". Tempo (in Indonesian). Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 Iqbal, Muhammad (14 February 2024). Dewi, Clara Maria Tjandra (ed.). "Banjir Rendam TPS di Kompleks Maharta Tangsel, Pencoblosan di Beberapa Lokasi Ditunda". Tempo. Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 Saputra, Dany (14 February 2024). "TPS dan Logistik Terdampak Banjir Warnai Pemilu 2024". kabar24.bisnis.com. Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 "Bawaslu Finds 19 Election Issues on Voting, Vote Counting Process". Tempo. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Police: 1,297 Polling Stations in Papua Yet to Conduct Voting". Tempo. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Polling Stations in West Java's Cimahi to Hold Follow-Up Election Over Empty Ballot Box". Tempo. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Bawaslu Confirms Rigged Ballot Papers in Bogor". Tempo. 15 February 2024. Archived from the original on 14 February 2024. Retrieved 15 February 2024.
 "Bawaslu Recommends a Re-vote in Kuala Lumpur". Tempo. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Bawaslu Investigates Allegations of Buying Ballots in Malaysia". Tempo. 27 February 2024. Archived from the original on 27 February 2024. Retrieved 27 February 2024.
 "Bawaslu records 1,200 violations during elections". Antara. 14 February 2024. Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 "Indonesia Election: 57 Poll Workers Died as of February 17, Health Ministry Says". Tempo. 19 February 2024. Archived from the original on 19 February 2024. Retrieved 19 February 2024.
 "Bawaslu Welcomes Anyone to Audit Sirekap Amid Mounting Pressure". Tempo. 15 February 2024. Archived from the original on 17 February 2024. Retrieved 15 February 2024.
 "PDIP Resmi Tolak Sirekap dan Penundaan Rekapitulasi Kecamatan". CNN Indonesia (in Indonesian). 21 February 2024. Archived from the original on 21 February 2024. Retrieved 21 February 2024.
 Andryanto, S. Dian (19 February 2024). "Ragam Alasan KPU Tunda Rekapitulasi Suara Pemilu 2024". Tempo (in Indonesian). Archived from the original on 21 February 2024. Retrieved 21 February 2024.
 "Early results showing Prabowo on course for landslide win in Indonesia's presidential election, securing around 60% of votes cast". CNA. 14 February 2024. Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 "Berita Acara dan Sertifikat Rekapitulasi Hasil Penghitungan Perolehan Suara" (PDF). General Elections Commission. Archived (PDF) from the original on 26 March 2024. Retrieved 26 March 2024.
 "Prabowo-Gibran Unggul di Semua Gugus Pulau". Kompas.id. 14 February 2024. Archived from the original on 15 February 2024. Retrieved 19 February 2024.
 "DAFTAR 81 Nama Caleg Terpilih di DPRA Aceh Periode 2024-2029, Hasil Rekapitulasi Suara Pemilu 2024". tribunnews.com (in Indonesian). Archived from the original on 16 March 2024.
 "Daftar 100 Anggota DPRD Sumut Terpilih Hasil Rekapitulasi KPU Provinsi". detik.com (in Indonesian). Archived from the original on 15 March 2024.
 Yenti, Juni Fitra. "Daftar 65 Nama Anggota DPRD Sumbar Terpilih 2024-2029 Hasil Rekapitulasi KPU Provinsi". sumbarkita.id (in Indonesian). Archived from the original on 16 March 2024.
 "Pleno KPU Riau, ini Daftar Anggota DPRD Riau Hasil Pileg 2024". riauterkini.com (in Indonesian). Archived from the original on 15 March 2024.
 Hakim, Anil. "Ini Daftar 55 Anggota DPRD Provinsi Jambi Terpilih Periode 2024-2029 dan Jumlah Perolehan Suara, Empat Orang Diantaranya Anak Kepala Daerah". metrojambi.com (in Indonesian). Archived from the original on 15 March 2024.
 Putra, Roki Eka. "Pleno KPU, Ini 45 Anggota DPRD Provinsi Bengkulu bakal Terpilih". rri.co.id (in Indonesian). Archived from the original on 15 March 2024.
 Romadon. "Ini Daftar 75 Caleg DPRD Provinsi Sumsel Terpilih, Dari Anak Bupati Hingga Putri Eks Gubernur Sumsel". sumselupdate.com (in Indonesian). Archived from the original on 16 March 2024.
 Hamapu, Alumudin. "Daftar 45 Caleg DPRD Provinsi Kepri Terpilih Hasil Rekapitulasi KPU". detik.com (in Indonesian). Archived from the original on 15 March 2024.
 "PDI-P Masih Dominasi Kursi DPRD Bangka Belitung". kompas.com (in Indonesian). Archived from the original on 20 March 2024.
 Saputra, Tommy (9 March 2024). "Gerindra Menang di Pileg DPRD Lampung, Raih 16 Kursi". detik.com (in Indonesian). Archived from the original on 10 March 2024. Retrieved 15 March 2024.
 "CATAT! 100 Caleg Terpilih DPRD Provinsi Banten Periode 2024-2029". bantenraya.com (in Indonesian). 15 March 2024. Archived from the original on 15 March 2024. Retrieved 15 March 2024.
 "Geser PDIP, PKS Raih Kursi Terbanyak di DPRD DKI Periode 2024-2029". metrotvnews.com (in Indonesian). Archived from the original on 16 March 2024.
 "KEPUTUSAN KOMISI PEMILIHAN UMUM PROVINSI JAWA BARAT NOMOR 13 TAHUN 2024 TENTANG PENETAPAN HASIL PEMILIHAN UMUM ANGGOTA DEWAN PERWAKILAN RAKYAT DAERAH PROVINSI JAWA BARAT TAHUN 2024" (PDF) (in Indonesian). General Elections Commission. Archived (PDF) from the original on 7 April 2024. Retrieved 7 April 2024.
 "Gedung DPRD Jateng Dikuasai PDIP, 33 Kader Duduk, Berikut 120 Nama Anggota Dewan Terpilih 2024". tribunnews.com (in Indonesian). Archived from the original on 17 March 2024.
 "PDIP Kuasai Kursi DPRD DIY, Gerindra 8, PKS 7, & PSI 1 Kursi". tirto.id (in Indonesian). Archived from the original on 16 March 2024.
 "Rekapitulasi KPU Final, Berikut 120 Calon Anggota DPRD Jatim Terpilih 2024 - 2029". kominfo.jatimprov.go.id (in Indonesian). Archived from the original on 15 March 2024.
 Samudero, Rizki Setyo (10 March 2024). "Ini Daftar 55 Anggota DPRD Bali 2024-2029, PDIP Masih Mendominasi". detik.com (in Indonesian). Archived from the original on 10 March 2024. Retrieved 15 March 2024.
 Akbar, Helmy (13 March 2024). "65 Caleg Terpilih DPRD NTB 2024-2029". detik.com (in Indonesian). Archived from the original on 15 March 2024. Retrieved 15 March 2024.
 Nedabang, Alfons (12 March 2024). "65 Anggota DPRD NTT Hasil Pemilu 2024, PDIP Golkar Gerindra 9 Kursi, 26 Wajah Baru dan 15 Srikandi". kupang.tribunnews.com (in Indonesian). Archived from the original on 12 March 2024. Retrieved 15 March 2024.
 "Daftar 65 Calon Anggota DPRD Provinsi Kalimantan Barat Terpilih 2024-2029". detik.com (in Indonesian). Archived from the original on 15 March 2024.
 Haryanto. "29 Wajah Baru Duduki Kursi DPRD Kalteng, Berikut Daftar Nama Lengkap Caleg Terpilih Hasil Pileg 2024". kalteng.tribunnews.com (in Indonesian). Archived from the original on 16 March 2024.
 Madrosid (9 March 2024). "DAFTAR Nama 55 Caleg Terpilih DPRD Kalsel Periode 2024-2029, Hasil Rapat Pleno Propinsi Pemilu 2024". pontianak.tribunnews.com (in Indonesian). Archived from the original on 9 March 2024. Retrieved 15 March 2024.
 "Peraih Kursi DPRD Kaltim Periode 2024-2029". beritakaltim.co (in Indonesian). Archived from the original on 16 March 2024. Retrieved 16 March 2024.
 "Peta Hasil Pemilu Kaltara". korankaltara.com (in Indonesian). Archived from the original on 17 March 2024.
 Alim, Sahrul. "Daftar 85 Caleg DPRD Sulawesi Selatan Terpilih, NasDem 'Panen' 17 Kursi". detik.com (in Indonesian). Archived from the original on 17 March 2024. Retrieved 15 March 2024.
 Syah, Adrian (12 March 2024). "Ini Daftar Nama dan Perolehan Suara 45 Anggota DPRD Sulbar Terpilih Periode 2024-2029". sulbaronline.com (in Indonesian). Archived from the original on 15 March 2024. Retrieved 15 March 2024.
 Admin HS (13 March 2024). "45 Anggota DPRD Provinsi Terpilih 2024-2029, NasDem Rebut Ketua DPRD". haluansultra.id (in Indonesian). Archived from the original on 15 March 2024. Retrieved 15 March 2024.
 Towengke, Simson (14 March 2024). "Hasil Pleno KPU, Berikut 55 Caleg DPRD Sulteng Dipastikan Lolos". posoonline.com (in Indonesian). Archived from the original on 17 March 2024.
 Satia (12 March 2024). "Daftar 45 Nama Anggota DPRD Gorontalo Periode 2024-2029, Banyak Pendatang Baru, Golkar 9 Kursi". tribunnews.com (in Indonesian). Archived from the original on 17 March 2024.
 Licin, Arham (11 March 2024). "Pleno KPU Berakhir, Berikut Daftar Anggota DPRD Provinsi Sulawesi Utara Periode 2024-2029". journaltelegraf.pikiran-rakyat.com (in Indonesian). Archived from the original on 15 March 2024. Retrieved 15 March 2024.
 Sardi, Sansul. "Daftar Lengkap 45 Anggota DPRD Maluku Utara Periode 2024-2029 Terpilih, Golkar Dominan". ternate.tribunnews.com (in Indonesian). Archived from the original on 17 March 2024. Retrieved 15 March 2024.
 "KPU Tetapkan 45 Nama Caleg DPRD Provinsi Maluku Terpilih". tribunnews.com (in Indonesian). Archived from the original on 19 March 2024.
 Weking, Fransiskus Salu. "KPU Papua Barat umumkan 35 calon anggota DPRD provinsi terpilih". papuabarat.antaranews.com (in Indonesian). Archived from the original on 15 March 2024.
 "Keputusan KPU Provinsi Papua Nomor 78 Tahun 2024 tentang Penetapan Hasil Pemilihan Umum Anggota Dewan Perwakilan Rakyat Papua Tahun 2024" (PDF) (in Indonesian). KPU RI. Archived (PDF) from the original on 27 March 2024. Retrieved 5 April 2024.
 "Keputusan KPU Provinsi Papua Tengah Nomor 37 Tahun 2024 tentang Penetapan Hasil Pemilihan Umum Anggota Dewan Perwakilan Rakyat Papua Tengah Tahun 2024" (PDF) (in Indonesian). KPU RI. Archived (PDF) from the original on 26 March 2024. Retrieved 5 April 2024.
 "Keputusan KPU Provinsi Papua Pegunungan Nomor 6 Tahun 2024 tentang Penetapan Hasil Pemilihan Umum Anggota Dewan Perwakilan Rakyat Daerah Provinsi Papua Pegunungan Tahun 2024" (PDF) (in Indonesian). KPU RI. Archived (PDF) from the original on 25 March 2024. Retrieved 5 April 2024.
 "KPU Tetapkan 35 Legislator DPR PBD Periode 2024-2029, Ini Daftar Lengkapnya". koreri.com (in Indonesian). Archived from the original on 18 March 2024.
 "11 Parpol tempati DPR Papsel, PDIP diprediksi pegang palu". jubi.id (in Indonesian). Archived from the original on 17 March 2024. Retrieved 17 March 2024.
 "Prabowo Delivers Victory Speech: Grateful for the Safe Election". Tempo. 14 February 2024. Archived from the original on 14 February 2024. Retrieved 15 February 2024.
 Karmini, Niniek; Tarigan, Edna (14 February 2024). "Indonesian defense chief linked to past rights abuses claims victory in presidential election". Associated Press. Archived from the original on 14 February 2024. Retrieved 14 February 2024.
 "Prabowo Subianto claims victory in Indonesian presidential election". Al Jazeera. 14 February 2024. Archived from the original on 14 February 2024. Retrieved 15 February 2024.
 "Indonesian activists protest ex-general's win in presidential election and allege massive fraud". Associated Press. 16 February 2024. Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 "Defeated Indonesian election candidates call for parliamentary probe". Reuters. 20 February 2024. Retrieved 21 February 2024.
 "Indonesia's Prabowo Subianto confirmed as president after first-round win". France 24. 21 March 2024. Retrieved 21 March 2024.
 Wee, Rae (15 February 2024). "Indonesian markets cheer as Prabowo's likely victory removes uncertainty". Reuters. Retrieved 21 February 2024.
 Sukma, Anshary Madya (14 February 2024). "National Police Chief Allows Crowds to Take to Streets to Protest 2024 Election Results as Long as They Are Orderly". kabar24.bisnis.com (in Indonesian). Archived from the original on 15 February 2024. Retrieved 14 February 2024.
 "Activists protest Prabowo's victory claims". NHK. 16 February 2024. Archived from the original on 17 February 2024. Retrieved 16 February 2024.
 "Indonesian activists protest ex-general's win in presidential election and allege massive fraud". ABC News. 16 February 2024. Archived from the original on 21 February 2024. Retrieved 21 February 2024.
 "Indonesia's defense minister, accused of abuses under dictatorship, is declared election winner". Associated Press. 20 March 2024. Retrieved 20 March 2024.
 "Anomali Suara PDIP Vs Ganjar di Quick Count, Ini Analisis LSI Denny JA". detikjogja (in Indonesian). 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Penyebab Ganjar-Mahfud Kalah di "Kandang Banteng"". KOMPAS.com (in Indonesian). 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 Setiawan, Bambang (15 February 2024). "Prabowo-Gibran Unggul di Semua Gugus Pulau". Kompas (in Indonesian). Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 Abshar-Abdalla, Ulil (14 February 2024). "Memahami Kemenangan Prabowo". Kompas (in Indonesian). Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Daftar 'Anomali' Pemilu: Ganjar Kalah di Jateng Hingga Gerindra Mandek". CNN Indonesia (in Indonesian). 17 February 2024. Archived from the original on 26 February 2024. Retrieved 26 February 2024.
 "Efek Jokowi: Kenapa Prabowo-Gibran menang dalam hitung cepat di lumbung suara PDI-P?". BBC News Indonesia (in Indonesian). 16 February 2024. Retrieved 20 March 2024.
 "Prabowo-Gibran Menang di "Kandang Banteng", TKN: Fenomenal Sekali". KOMPAS.com (in Indonesian). 15 March 2024. Retrieved 20 March 2024.
 "Pemilu 2024 vs 2019: PDIP Turun 6%, Golkar Terbang 34%, PKS Naik 11%". CNBC Indonesia (in Indonesian). 22 March 2024. Retrieved 27 March 2024.
 "Pemerhati Ungkap Alasan Suara PDIP Turun Drastis". RRI (in Indonesian). 23 March 2024. Retrieved 13 April 2024.
 "Pemilu 2024: Mengapa PPP gagal lolos ke parlemen? - 'Lemahnya kaderisasi, peristiwa naturalisasi Sandiaga, hingga ideologi'". BBC News Indonesia (in Indonesian). 22 March 2024. Retrieved 27 March 2024.
 Safitri, Eva (15 February 2024). "Jokowi Ngaku Sudah Bertemu Prabowo-Gibran, Ucapkan Selamat". detiknews (in Indonesian). Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 Al Ayyubi, Sholahuddin (17 February 2024). "SBY ke Prabowo Subianto: Beliau Komandan Saya Sekarang". Bisnis.com (in Indonesian). Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 "Surya Paloh: NasDem Terima Hasil Pemilu 2024, Selamat Prabowo-Gibran". CNN Indonesia (in Indonesian). Retrieved 21 March 2024.
 Mawangi, Genta Tenri (20 March 2024). "NasDem terima hasil Pemilu 2024, ucapkan selamat untuk Prabowo-Gibran". Antara (in Indonesian). Retrieved 21 March 2024.
 "PKS Terima Hasil Pileg dan Pilpres 2024: Masalah Hukum Lain Cerita". CNN Indonesia (in Indonesian). Retrieved 21 March 2024.
 Fauzi, Fauzi (20 March 2024). "Anies-Muhaimin sampaikan sikap politik hasil Pilpres 2024". Antara (in Indonesian). Retrieved 21 March 2024.
 Darwati, Erta (20 March 2024). "Anies-Muhaimin Siap Gugat ke MK, Tolak Hasil Pilpres 2024 yang Ditetapkan KPU". Bisnis.com (in Indonesian). Retrieved 21 March 2024.
 Laoh, Gisella Previan. "TPN Bakal Gugat ke MK pada H+3 Pengumuman Hasil Pilpres oleh KPU". detiknews (in Indonesian). Retrieved 21 March 2024.
 Rizaldi, Bagus Ahmad (21 March 2024). "Romahurmuziy: PPP tolak hasil rekapitulasi suara Pemilu 2024". Antara News (in Indonesian). Retrieved 21 March 2024.
 Anggrainy, Firda Cynthia. "Imam Besar Istiqlal Beri Selamat ke Prabowo: Semoga Indonesia Makin Jaya". detiknews (in Indonesian). Retrieved 21 March 2024.
 Fauziah, Anisa. "Prabowo Gibran Menang Pilpres 2024, PBNU Ucapkan Selamat". beritasatu.com (in Indonesian). Retrieved 21 March 2024.
 Lestari, Dedy Priatmojo, Yeni (22 March 2024). "PP Muhammadiyah Beri Selamat ke Prabowo-Gibran, Hormati Pihak yang Menggugat ke MK". www.viva.co.id (in Indonesian). Retrieved 22 March 2024.
 Kusumo, Herlambang Jati (22 March 2024). "Sri Sultan HB X Beri Ucapan Selamat ke Prabowo-Gibran". IDN Times (in Indonesian). Retrieved 22 March 2024.
 "Prabowo terima kunjungan Dubes China". Antara News (in Indonesian). 18 February 2024. Archived from the original on 18 February 2024. Retrieved 18 February 2024.
 "Xi congratulates Prabowo Subianto on election as Indonesian president-Xinhua". Xinhua. Retrieved 21 March 2024.
 "China's Xi Congratulates Indonesia's Prabowo On Election Win". Barron's. Retrieved 21 March 2024.
 Subianto, Prabowo [@prabowo] (22 March 2024). "Terima kasih saya ucapkan kepada Presiden Republik Rakyat Tiongkok, Yang Mulia Xi Jinping atas ucapan selamat terhadap hasil akhir Pemilu yang diantarkan langsung oleh Duta Besar Tiongkok untuk Indonesia, H.E @Amb_LuKang" (Tweet). Retrieved 22 March 2024 – via Twitter.
 Sanches, Hortencio (19 February 2024). "Horta kongratula Prabowo Subianto ne'ebé hetan vitória iha eleisaun presidensiál". Tatoli (in Tetum). Archived from the original on 21 February 2024. Retrieved 21 February 2024.
 "PM India Modi Beri Selamat ke Prabowo, Singgung Kerja Sama Strategis". CNN Indonesia (in Indonesian). Archived from the original on 19 February 2024. Retrieved 19 February 2024.
 "Prabowo Terima Ucapan Selamat Hasil Sementara Pilpres dari Presiden Iran". detik.com (in Indonesian). 9 March 2024. Archived from the original on 8 March 2024. Retrieved 9 March 2024.
 Nugraha, Fajar (21 March 2024). "Perdana Menteri Jepang Fumio Kishida Ucapkan Selamat ke Prabowo Subianto". medcom.id (in Indonesian). Retrieved 21 March 2024.
 Marison, Walda (12 March 2024). "Raja Abdullah II Yordania beri ucapan selamat Prabowo". Antara (in Indonesian). Archived from the original on 12 March 2024. Retrieved 12 March 2024.
 Nurganingsih, Sri. "Prabowo Terima 4 Ucapan Selamat dari Pemimpin Dunia, TKN: Pertanda Demokrasi". RM.ID (in Indonesian). Archived from the original on 16 February 2024. Retrieved 19 February 2024.
 Ibrahim, Anwar [@anwaribrahim] (20 March 2024). "Saya telah menghubungi Presiden Terpilih Republik Indonesia Yang Mulia Bapak Prabowo Subianto bagi menyampaikan salam tahniah di atas kejayaan beliau dalam Pemilihan Presiden Indonesia yang baru selesai. Saya dimaklumkan saya pemimpin pertama yang menyampaikan ucapan tahniah kepada beliau selepas pengumuman keputusan rasmi PEMILU sebentar tadi. Ini cukup bermakna kerana melambangkan nilai persahabatan antara Malaysia dan Indonesia yang amat istimewa sebagai negara tetangga yang dekat dan penting" (Tweet). Retrieved 21 March 2024 – via Twitter.
 "Prabowo terima ucapan selamat presiden Palestina karena unggul suara". Antara (in Indonesian). 7 March 2024. Archived from the original on 7 March 2024. Retrieved 17 February 2024.
 "Prabowo Subianto Terima Ucapan Selamat dari Presiden Palestina Mahmoud Abbas". Sindonews (in Indonesian). 7 March 2024. Archived from the original on 7 March 2024. Retrieved 17 February 2024.
 Meilanova, Denis (15 February 2024). "Bongbong Marcos Beri Selamat ke Prabowo, Nantikan Penguatan Kemitraan RI-Filipina". Bisnis.com (in Indonesian). Archived from the original on 20 February 2024. Retrieved 21 February 2024.
 Bajo, Anna Felicia (22 February 2024). "Marcos wants strengthened ties with Indonesia under new presumptive president Prabowo". GMA Integrated News. Archived from the original on 28 February 2024. Retrieved 28 February 2024.
 "President Marcos congratulates new Indonesian president". The Philippine Star. Retrieved 28 February 2024.
 "Custodian of the Two Holy Mosques Congratulates Prabowo Subianto on Winning Presidential Elections of Indonesia". Saudi Press Agency. 21 March 2024. Retrieved 21 March 2024.
 "HRH Crown Prince Congratulates Prabowo Subianto on Winning Presidential Elections of Indonesia". Saudi Press Agency. 21 March 2024. Retrieved 21 March 2024.
 Woon, Wallace (21 March 2024). "President Tharman, PM Lee congratulate Indonesia's President-elect Prabowo". The Straits Times. ISSN 0585-3923. Retrieved 21 March 2024.
 "PM Lee congratulates Prabowo after strong performance in Indonesia presidential election". CNA. 15 February 2024. Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 "Yoon holds phone talks with Indonesia's president-elect". The Korea Times. Retrieved 17 April 2024.
 "KPU Mengumumkan Hasil Pemilu Indonesia 2024, Presiden Tsai Ing-wen Memberikan Ucapan Selamat Atas Terpilihnya Prabowo Subianto Sebagai Presiden Baru Indonesia". Radio Taiwan International (in Indonesian). Retrieved 21 March 2024.
 "Perdana Menteri Thailand Beri Selamat kepada Prabowo atas Kemenangan Pemilu -". Tempo.co. Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 Safitri, Eva. "Erdogan Beri Selamat ke Prabowo Lewat Surat: Mr President Elect, Dear Brother". Detik.com (in Indonesian). Archived from the original on 22 February 2024. Retrieved 23 February 2024.
 "Prabowo Terima Telepon Erdogan, Dapat Ucapan Menang Pilpres". news.detik.com (in Indonesian). Retrieved 10 April 2024.
 Hutajulu, Matius (21 February 2024). "Prabowo Terima Ucapan Selamat dari Presiden MBZ: Cermin Persahabatan RI-UEA". Detik.com (in Indonesian). Archived from the original on 21 February 2024. Retrieved 16 February 2024.
 Maulana, Gibran (15 February 2024). "PM Australia Gercep Telepon Prabowo yang Unggul QC, Ini yang Dibahas". detiknews (in Indonesian). Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 @AlboMP (15 February 2024). "I was honoured to be the first foreign leader to speak today with @prabowo, who has a clear lead in official and unofficial counts, about my ambition for the future of Australia – Indonesia relations" (Tweet) – via Twitter.
 Fadilah, Kurniawan (23 February 2024). "Wakil PM Australia Beri Selamat ke Prabowo: Anda Presiden RI Selanjutnya". detiknews (in Indonesian). Archived from the original on 23 February 2024. Retrieved 23 February 2024.
 Peters, Winston [@NewZealandMFA] (20 March 2024). "Congratulations Prabowo Subianto on your success in Indonesia's presidential elections, as announced overnight" (Tweet). Retrieved 21 March 2024 – via Twitter.
 Prasetyo, Noor Arief (15 March 2024). "Menteri Luar Negeri Selandia Baru Ucapkan Selamat ke Prabowo Subianto atas Hasil Pemilu". harian.disway.id (in Indonesian). Retrieved 21 March 2024.
 Setiawan, Agus (15 February 2024). "PM Ceko Beri Ucapan Selamat kepada Prabowo-Gibran, Nyatakan Siap Perkuat Hubungan Bilateral". viva.co.id (in Indonesian). Archived from the original on 15 February 2024. Retrieved 15 February 2024.
 Baskoro, Faisal Maliki (9 March 2024). "France's Macron Congratulates Prabowo on Election Win". Jakarta Globe. Archived from the original on 9 March 2024. Retrieved 9 March 2024.
 Pramudyani, Yashinta Difa (22 March 2024). "Kanselir Jerman ucapkan selamat kepada Prabowo". Antara News (in Indonesian). Retrieved 22 March 2024.
 Sorongan, Tommy Patrio. "PM Eropa Teman Putin Ucapkan Selamat ke Prabowo, Beri Pesan Ini". CNBC Indonesia (in Indonesian). Retrieved 22 March 2024.
 Yoanita, Djohan (15 February 2024). "PM Belanda Rutte ucapkan selamat pada Prabowo atas keunggulannya". Antara (in Indonesian). Archived from the original on 15 February 2024. Retrieved 16 February 2024.
 "Putin congratulates Indonesian President-elect Subianto with victory at polls — Kremlin". TASS. 16 February 2024. Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 @buducnostsrbijeav (22 February 2024). "Predsednik Republike Srbije Aleksandar Vučić razgovarao je telefonom sa ministrom odbrane i predsedničkim kandidatom Indonezije Prabovom Subiantom" – via Instagram.
 Haryadi, Malvyandie. "Perdana Menteri Spanyol Ucapkan Selamat ke Prabowo atas Keunggulan di Pilpres via Surat Resmi". tribunnews.com. Archived from the original on 16 March 2024. Retrieved 15 March 2024.
 "Presiden Swiss Ucapkan Selamat ke Prabowo, Ini Pesan Lengkapnya". cnbcindonesia.com (in Indonesian). Retrieved 21 March 2024.
 Court, Elsa (18 March 2024). "Zelensky holds call with Indonesian President-elect Subianto, extends Peace Summit invitation". The Kyiv Independent. Archived from the original on 18 March 2024. Retrieved 18 March 2024.
 Azzahra, Tiara Aliya. "Dubes Inggris Temui Prabowo, Beri Surat Ucapan Selamat dari Rishi Sunak". Detik.com (in Indonesian). Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 "British PM Rishi Sunak Congratulates Prabowo on Election Success". Jakarta Globe. Archived from the original on 17 February 2024. Retrieved 17 February 2024.
 Arismunandar, Satrio. "Ucapan Selamat Presiden Nikaragua Daniel Ortega Saavedra dan Wapres Rosario Murillo Kepada Prabowo Subianto". ORBITINDONESIA.COM. Archived from the original on 23 February 2024. Retrieved 23 February 2024.
 Consejo de Comunicación y Ciudadanía (16 February 2024). "Gobierno de Nicaragua saluda al Hermano Prabowo Subianto, Presidente de Indonesia". El 19 Digital (in Spanish). Archived from the original on 13 March 2024. Retrieved 13 March 2024.
 "Election in Indonesia PRESS STATEMENT". United States Department of State. 14 February 2024. Archived from the original on 15 February 2024. Retrieved 14 February 2024.
 "Daftar Pemimpin Dunia Ucapkan Selamat ke Prabowo: Rusia Inggris hingga China, AS Agak Lain". Tribunnews.com (in Indonesian). 18 February 2024. Archived from the original on 18 February 2024. Retrieved 18 February 2024.
 Ibrahim, Gibran Maulana (14 March 2024). "Joe Biden Surati Prabowo, Beri Ucapan Selamat unggul Pilpres 2024". news.detik.com (in Indonesian). Archived from the original on 14 March 2024. Retrieved 14 March 2024.
 Jati, Haryo (23 March 2024). "Joe Biden Akhirnya Telepon Presiden Terpilih Indonesia Prabowo untuk Beri Selamat, Ini Katanya". Kompas TV (in Indonesian). Retrieved 23 March 2024.
 Wijayaatmaja, Yakub Pryatama (20 March 2024). "Menlu AS Ucapkan Selamat ke Prabowo Sebagai Presiden Terpilih". mediaindonesia.com (in Indonesian). Retrieved 21 March 2024.
 Blinken, Antony [@SecBlinken] (20 March 2024). "Congratulations to President-elect @Prabowo Subianto on his victory in Indonesia's Presidential Election. We look forward to partnering closely with the President-elect and his Administration when they take office in October" (Tweet). Retrieved 21 March 2024 – via Twitter.
 Zamani, Labib; Rusiana, Dita Angga (20 March 2024). "Tak Hanya Prabowo, Gibran Juga Dapat Ucapan Selamat Unggul di Pilpres dari Pejabat Negara Sahabat". KOMPAS.com (in Indonesian). Retrieved 22 March 2024.
External links
Presidential vote count (Archived 14 February 2024 at the Wayback Machine) at the General Elections Commission (KPU) (in Indonesian)
vte
2024 Indonesian general election
vte
Indonesia Elections in Indonesia
Categories: 2024 elections in IndonesiaFebruary 2024 events in IndonesiaLegislative elections in IndonesiaPresidential elections in Indonesia
This page was last edited on 18 April 2024, at 21:46 (UTC).
Text is available under the Creative Commons Attribution-ShareAlike License 4.0; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.
Privacy policyAbout WikipediaDisclaimersContact WikipediaCode of ConductDevelopersStatisticsCookie statementMobile view
Wikimedia FoundationPowered by MediaWiki
Toggle limited content width
 Kiley, Bill [@Billtron209] (July 12, 2019). "Indie game fun fact: I purchased my studio monitors from @Disasterpeace years ago when he upgraded. This means that the music I wrote for Katana ZERO and the OST from FEZ was composed on the same physical speakers. Small music world?" (Tweet). Archived from the original on October 21, 2021. Retrieved February 11, 2022 – via Twitter.
 Williams, Mike (September 8, 2015). "Katana Zero Will Make You Feel Like a Badass". USgamer. Archived from the original on February 10, 2022. Retrieved February 10, 2022.
 Priestman, Chris (December 19, 2015). "Katana Zero Will Turn You Into A Time-Manipulating Assassin In 2016". Siliconera. Archived from the original on February 10, 2022. Retrieved February 10, 2022.
 Devore, Jordan (December 17, 2015). "Katana Zero has the right stuff". Destructoid. Archived from the original on February 10, 2022. Retrieved February 10, 2022.
 Wong, Alistar (June 18, 2016). "Time-Manipulating Assassin Game Katana Zero Slips To 2017". Siliconera. Archived from the original on August 7, 2022. Retrieved August 7, 2022.
 Devore, Jordan (January 11, 2019). "It's almost time for the bullet-slashing action game Katana Zero". Destructoid. Archived from the original on February 8, 2022. Retrieved February 12, 2022.
 LeChair, Kyle (January 17, 2019). "Devolver Digital to Publish Katana Zero". Hardcore Gamer. Archived from the original on August 7, 2022. Retrieved August 7, 2022.
 LeClair, Kyle (March 20, 2019). "Katana Zero Reveals Release Date in New Trailer". Hardcore Gamer. Archived from the original on February 11, 2022. Retrieved February 10, 2022.
 Bolding, Jonathan (February 15, 2020). "Katana Zero's free DLC will be three times larger than expected". PC Gamer. Archived from the original on February 12, 2022. Retrieved February 12, 2022.
 Doolan, Liam (April 18, 2019). "Katana Zero Is Devolver Digital's Most Pre-Ordered Switch Game So Far". Nintendo Life. Archived from the original on February 12, 2022. Retrieved February 12, 2022.
 Walker, Alex (April 18, 2019). "Katana Zero Just Got Banned In Australia". Kotaku. Archived from the original on April 17, 2019. Retrieved February 8, 2022.
 Doolan, Liam (April 17, 2019). "Katana Zero Refused Classification In Australia And New Zealand". Nintendo Life. Archived from the original on October 13, 2022. Retrieved October 13, 2022.
 Walker, Alex (May 16, 2019). "Katana Zero's Ban Overturned, Receives R18+ Rating". Kotaku. Archived from the original on April 2, 2020. Retrieved February 8, 2022.
 Romano, Sal (October 13, 2020). "Xbox Game Pass adds Tales of Vesperia: Definitive Edition, Katana Zero, Supraland, and more in late October". Gematsu. Archived from the original on January 28, 2022. Retrieved October 16, 2022.
 Diaz, Justin (July 28, 2022). "Amazon Luna: Everything You Need To Know – Updated July 28, 2022". Android Authority. Archived from the original on August 31, 2022. Retrieved July 29, 2022.
 Romano, Sal (March 1, 2021). "ESRB rates Katana ZERO for PS4, Demon Turf for PS5". Gematsu. Archived from the original on March 1, 2021. Retrieved September 10, 2022.
 Romano, Sal (September 2, 2022). "Katana ZERO DLC first gameplay footage". Gematsu. Archived from the original on September 13, 2022. Retrieved September 10, 2022.
 LeBlanc, Wesley (13 November 2023). "Hades, Katana Zero, Death's Door, And More Coming To Netflix Soon". Game Informer. Retrieved 4 December 2023.
 Doolan, Liam (April 26, 2019). "Katana Zero Sold More Than 100,000 Copies In Its First Week". Nintendo Life. Archived from the original on April 26, 2019. Retrieved April 26, 2019.
 Stander; Justin (January 16, 2020). Katana Zero Creator Justin Stander On The Relentless Drive That Brought The Game To Life. Forbes. Archived from the original on February 15, 2022. Retrieved February 12, 2022 – via YouTube.
 "Katana Zero for Switch Reviews". Metacritic. Archived from the original on May 5, 2019. Retrieved February 8, 2022.
 "Katana Zero for PC Reviews". Metacritic. Archived from the original on April 29, 2019. Retrieved February 8, 2022.
 "Katana Zero Reviews". OpenCritic. 18 April 2019. Retrieved June 13, 2023.
 Juba, Joe (April 18, 2019). "Katana Zero Review - A Sharp And Dull Blade". Game Informer. Archived from the original on April 19, 2019. Retrieved March 15, 2022.
 Lloyd, David (April 18, 2019). "Katana Zero Review". Nintendo World Report. Archived from the original on February 8, 2022. Retrieved March 15, 2022.
 Frushtick, Russ (April 18, 2019). "Katana Zero review: a game about death, death, death, and life". Polygon. Archived from the original on March 31, 2022. Retrieved March 15, 2022.
 Dietz, Jason (January 3, 2020). "Best Video Games of 2019". Metacritic. Archived from the original on February 19, 2020. Retrieved April 6, 2022.
 Cryer, Hirun (December 27, 2019). "Hirun Cryer's Top 10 Games of 2019: Excuse the Turmoil". USgamer. Archived from the original on February 12, 2022. Retrieved April 6, 2022.
 Khan, Joshua (December 30, 2019). "The 30 Best Video Games of 2019". Thrillist. Archived from the original on February 13, 2022. Retrieved April 6, 2022.
 Sillis, Ben; Hunt-Stevenson, Jamie (December 3, 2019). "These are the 10 best games of 2019". Red Bull. Archived from the original on February 14, 2022. Retrieved April 6, 2022.
 "The Best Action Game of 2019". IGN. December 10, 2019. Archived from the original on February 12, 2022. Retrieved April 6, 2022.
 "The Best Video Game Music/Soundtrack of 2019". IGN. December 10, 2019. Archived from the original on October 21, 2021. Retrieved April 6, 2022.
 Trent, Logan (January 29, 2019). "Announcing the 2019 SXSW Gamer's Voice Award Nominees!". South by Southwest. Archived from the original on April 12, 2019. Retrieved January 22, 2020.
 Winslow, Jeremy (November 19, 2019). "The Game Awards 2019 Nominees Full List". GameSpot. Archived from the original on November 23, 2019. Retrieved November 21, 2019.
 Lee, Jess (January 7, 2020). "Independent Games Festival: Mutazione leads nominations for 2020 awards". Digital Spy. Archived from the original on July 15, 2021. Retrieved October 16, 2022.
 Blake, Vikki (February 18, 2020). "Here are the nominees for SXSW Gaming Awards 2020". MCV/Develop. Archived from the original on November 16, 2021. Retrieved April 8, 2022.
 Stuart, Keith (March 3, 2020). "Death Stranding and Control dominate Bafta games awards nominations". The Guardian. Archived from the original on March 3, 2020. Retrieved March 4, 2020.
 Tack, Daniel (April 25, 2019). "Katana Zero Is Getting Free DLC And A Speedrun Mode". Game Informer. Archived from the original on February 20, 2022. Retrieved February 19, 2022.
 Doolan, Liam (March 26, 2021). "Katana Zero Dev Says Free DLC Is Now So Big It's "More Like Katana 1.5"". Nintendo Life. Archived from the original on February 20, 2022. Retrieved February 19, 2022.
 Doolan, Liam (February 16, 2020). "Katana Zero Free DLC Still In Development And It's Now "3x The Original Planned Size"". Nintendo Life. Archived from the original on February 20, 2022. Retrieved February 19, 2022.
Works cited
Barnes, Morgan (May 2, 2019). "LudoWic - Katana Zero Composer". In the Blood (Podcast). Archived from the original on September 17, 2022. Retrieved October 14, 2021 – via Spotify.
"Top 47,858 Games of All Time Episode 351: Katana Zero". Hardcore Gaming 101 (Podcast). November 21, 2020. Archived from the original on November 18, 2021. Retrieved October 16, 2022.
External links
	Video games portal
icon	2010s portal
Official website
Categories: 2019 video gamesAction gamesCyberpunk video gamesDevolver Digital gamesDystopian video gamesGameMaker Studio gamesHack and slash gamesIndie gamesMacOS gamesNeo-noir video gamesNintendo Switch gamesOrganized crime video gamesPlatformersScience fiction video gamesSingle-player video gamesVideo games about ninjaVideo games developed in the United StatesWindows gamesXbox One games
This page was last edited on 18 April 2024, at 19:30 (UTC).
Text is available under the Creative Commons Attribution-ShareAlike License 4.0; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.
Privacy policyAbout WikipediaDisclaimersContact WikipediaCode of ConductDevelopersStatisticsCookie statementMobile view
Wikimedia FoundationPowered by MediaWiki
Toggle limited content width`


let text = 'Hello There';
const corpus = corpusText.match(/'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu)



const bpeTokenizer = new BPETokenizer();
bpeTokenizer.learnVocab(corpus, 20);
bpeTokenizer.purgeVocabulary(5000); 



// Function for normalization between min and max
function normalizeBetweenMinMax(array, min, max) {
  const normalizedArray = array.map(value => (value - min) / (max - min));
  return normalizedArray;
}

// Function for denormalization between min and max
function denormalizeBetweenMinMax(normalizedArray, min, max) {
  const denormalizedArray = normalizedArray.map(value => value * (max - min) + min);
  return denormalizedArray;
}

// Example usage:
const numbers = [200, 500, 750, 1000];
const min = 0;
const max = 2000;

// Normalize the numbers between min and max
const normalizedNumbers = normalizeBetweenMinMax(bpeTokenizer.tokenize(text).tokenIDs, min, max);
console.log("Normalized numbers:", normalizedNumbers);

// Denormalize the normalized numbers back to the original range
const denormalizedNumbers = denormalizeBetweenMinMax(normalizedNumbers, min, max);
console.log("Denormalized numbers:", denormalizedNumbers);


const axios = require('axios');

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows)
            .fill()
            .map(() => Array(this.cols).fill(0));
    }

    randomize() {
        this.data = this.data.map((row) => row.map(() => Math.random() * 2 - 1)); // Random numbers between -1 and 1
    }

    static fromArray(arr) {
        return new Matrix(arr.length, 1).map((_, i) => arr[i]);
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    map(func) {
        this.data = this.data.map((row, i) => row.map((val, j) => func(val, i, j)));
        return this;
    }

    static map(matrix, func) {
        return new Matrix(matrix.rows, matrix.cols).map((_, i, j) => func(matrix.data[i][j], i, j));
    }

    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows).map((_, i, j) => matrix.data[j][i]);
    }

    static elementWiseProduct(a, b) {
        const rows = a.rows;
        const cols = a.cols;

        if (b.rows !== rows || b.cols !== cols) {
            throw new Error("Matrices dimensions don't match for element-wise multiplication");
        }

        const result = new Matrix(rows, cols);

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }

        return result;
    }

    static dot(a, b) {
        if (a.cols !== b.rows) {
            console.error('Columns of A must match rows of B for matrix multiplication.');
            return;
        }

        return new Matrix(a.rows, b.cols).map((_, i, j) => {
            // console.log(`Calculating element (${i}, ${j})`);
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                //  console.log(`   Adding ${a.data[i][k]} * ${b.data[k][j]}`);
                sum += a.data[i][k] * b.data[k][j];
            }
            return sum;
        });
    }

    static subtract(a, b) {
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }

    scalarMultiply(matrix, scalar) {
        return new Matrix(matrix.rows, matrix.cols).map((val) => val * scalar);
    }

    sum() {
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        return sum;
    }

    add(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val + matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val + matrix));
        }
        return this;
    }

    static multiply(a, b) {
        if (b instanceof Matrix) {
            // Hadamard product
            return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] * b.data[i][j]);
        } else {
            // Scalar product
            return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] * b);
        }
    }

    subtract(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val - matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val - matrix));
        }
        return this;
    }

    multiply(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val * matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val * matrix));
        }
        return this;
    }
    static concatenateVertical(a, b) {
        if (a.cols !== b.cols) {
            console.error('Number of columns must match for vertical concatenation.');
            return;
        }

        const resultRows = a.rows + b.rows;
        const resultCols = a.cols;
        const resultData = [];

        for (let i = 0; i < a.rows; i++) {
            resultData.push([...a.data[i]]);
        }

        for (let i = 0; i < b.rows; i++) {
            const row = [];
            for (let j = 0; j < b.cols; j++) {
                row.push(b.data[i][j]);
            }
            resultData.push(row);
        }

        const result = new Matrix(resultRows, resultCols);
        result.data = resultData;
        return result;
    }

    static multiplyElementWise(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.error('Dimensions must match for element-wise multiplication.');
            return;
        }

        const result = new Matrix(a.rows, a.cols);
        result.data = a.data.map((row, i) => row.map((val, j) => val * b.data[i][j]));
        return result;
    }
}

class Layer {
    constructor(inputSize, outputSize, activation) {
        this.weights = new Matrix(outputSize, inputSize);
        // Glorot initialization for weights
        const glorotFactor = Math.sqrt(2 / (inputSize + outputSize));
        this.weights.randomize(-glorotFactor, glorotFactor);
        this.bias = new Matrix(outputSize, 1);
        this.bias.randomize(); // Small random bias initialization
        this.activation = activation;
    }

    forward(input) {
        this.input = input;
        this.weightedSum = Matrix.dot(this.weights, input).add(this.bias);

        this.output = this.weightedSum.map(this.activation.func);

        return this.output;
    }

    backward(dOutput, target) {
        // Calculate the gradient of the loss with respect to the inputs of the layer
        const dInputs = Matrix.dot(Matrix.transpose(this.weights), dOutput);

        // Calculate the gradient of the loss with respect to the weights of the layer
        const dWeights = Matrix.dot(dOutput, Matrix.transpose(this.input));

        // Calculate the gradient of the loss with respect to the biases of the layer
        const dBias = dOutput;

        // Return the gradients
        return { dInputs, dWeights, dBias };
    }

    train(input, target, learningRate) {
        // Forward pass
        const output = this.forward(input);

        // Compute error
        let error;
        error = Matrix.subtract(target, output);

        // Compute gradients
        let gradients;
        gradients = this.weightedSum.map(this.activation.dfunc);
        gradients.multiply(error);
        gradients.scalarMultiply(learningRate);

        // Compute weight deltas
        const deltas = Matrix.dot(gradients, Matrix.transpose(input));

        // Update weights and biases
        this.weights.add(deltas);
        this.bias.add(gradients);

        // Compute and return Mean Squared Error (MSE)
        const sumSquaredErrors = error.map((x) => x * x).sum(); // Sum of squared errors
        const mse = sumSquaredErrors / error.rows; // Mean Squared Error

        return mse;
    }
}

const sigmoid = {
    func: (x) => 1 / (1 + Math.exp(-x)),
    dfunc: (y) => y * (1 - y), // derivative of sigmoid
};

const leakyRelu = {
    func: (x) => (x > 0 ? x : 0.01 * x), // Leaky ReLU with alpha = 0.01
    dfunc: (y) => (y > 0 ? 1 : 0.01), // derivative of Leaky ReLU
};
const swish = {
    func: (x) => x / (1 + Math.exp(-x)),
    dfunc: (y, x) => y + (1 - y / (1 + Math.exp(-x)))
};

const relu = {
    func: (x) => Math.max(0, x),
    dfunc: (y) => (y > 0 ? 1 : 0), // derivative of ReLU
};

const tanh = {
    func: (x) => Math.tanh(x),
    dfunc: (y) => 1 - y * y, // derivative of tanh
};
const softplus = {
    func: (x) => Math.log(1 + Math.exp(x)),
    dfunc: (y) => 1 / (1 + Math.exp(-y)), // derivative of softplus
};
const elu = {
    alpha: 0.01, // or any small positive value
    func: (x) => (x > 0 ? x : elu.alpha * (Math.exp(x) - 1)),
    dfunc: (y) => (y > 0 ? 1 : y + elu.alpha), // derivative of ELU
};
const gelu = {
    func: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
    dfunc: (x) => {
        const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
        return 0.5 * (1 + Math.pow(cdf, 2) * (1 - Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)) * Math.exp(-(Math.pow(x, 2) / 2))));
    },
};

function encodeStringToTokens(str) {
    // Split the string into an array of characters
    const chars = str.split('');

    // Map each character to its Unicode value and normalize it to a value between 0 and 1
    const encoded = chars.map((char) => char.charCodeAt(0) / 65535); // 65535 is the maximum Unicode value

    return encoded;
}

function decodeTokensToString(tokens) {
    // Map each token back to its corresponding character using Unicode values
    const decoded = tokens.map((token) => String.fromCharCode(Math.floor(token * 65535))); // 65535 is the maximum Unicode value

    // Join the characters into a string
    return decoded.join('');
}

// Example usage

hiddenLayer = new Layer(101, 4, gelu);
gate = new Layer(101,101,sigmoid);
gate2 = new Layer(101,101,swish);

let offset = 0;
const batchSize = 32;
const totalExamples = 700000; // Update this with the total number of examples in your dataset

function fetchData(offset) {
    axios
        .get('https://datasets-server.huggingface.co/rows', {
            params: {
                dataset: 'stingning/ultrachat',
                config: 'default',
                split: 'train',
                offset: offset,
                length: batchSize,
            },
        })
        .then((response) => {
            // Handle successful response
            for (let i = 0; i < 4; i++) {
                let g = 0;
                let lr = 0.001;
                response.data.rows.forEach((row) => {
                   
                    const inputString = row.row.data[0];
                  
                    const encodedTokens = normalizeBetweenMinMax(bpeTokenizer.tokenize(inputString).tokenIDs, 0, 5000);
                 

                    // Inside your loop
                    for (let i = 0; i < inputString.length - 20; i++) {
                        if (encodedTokens[i + 104] === undefined) {
                            break;
                        }
                        let input = Matrix.fromArray(encodedTokens.slice(i, i + 101));
                        let output = Matrix.fromArray([encodedTokens[i + 101],encodedTokens[i + 102],encodedTokens[i + 103],encodedTokens[i + 104]] || [0]);

                
                        for (let i = 0; i < 3; i++) {
                            gated = gate.forward(input)
                            gated2 = gate2.forward(input)
                            gate.train(input,test,lr)
                            gate2.train(input,test,lr)
                           
                            test = Matrix.multiplyElementWise(gated,gated2)
                            hiddenOutput = hiddenLayer.forward(test);

                           

                           error = hiddenLayer.train(test, output, lr);
                  

                            console.log('input', decodeTokensToString(input.toArray()));
                            console.log('output', decodeTokensToString(hiddenOutput.toArray()));
                            console.log(error);
                            lr -= 1e-6;
                            g++
                            console.log(lr)
                        }
                    }
                });
            }
            // Check if there are more examples to fetch
            if (offset + batchSize < totalExamples) {
                // Fetch the next batch of examples
                offset += batchSize;
                fetchData(offset);
            } else {
                console.log('Finished fetching all examples.');
            }
        


        })
        .catch((error) => {
            // Handle error
            console.error('Error fetching data:', error);
        });
      
}

// Start fetching data from offset 0
fetchData(offset);
