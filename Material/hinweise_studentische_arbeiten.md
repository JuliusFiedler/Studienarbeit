# Hinweise zur Anfertigung von Studien-, Diplom- und Masterarbeiten

Stand: 2019-10-30 16:10:49

Autoren:
    - Carsten Knoll



## Schreibprozess

- Gliederung anlegen und immer weiter verfeinern, zu einzelnen Abschnitten Stichpunkte aufschreiben, dann ausformulieren und ggf. andere Abschnitte mit Stichpunkten ergänzen
- Einleitung und Vorwort und Zusammenfassung und Ausblick ziemlich am Ende ausformulieren.
- Nicht die Zeit für das Schreiben unterschätzen. Realistischer Richtwert: 1-5 Seiten pro Tag.
- Ausreichend Zeit für Korrekturen einplanen (Zeitplan machen und regelmäßig überprüfen, spätestens eine Woche vor dem Ausdrucken sollte die Arbeit inhaltlich fertig sein.)
- Möglichst externe Korrekturleser organisieren (für die Sprache, für den Inhalt bzw. Verständlichkeit)
- nicht "beratungsresistent" sein, andere Entscheidungen ggf. begründen
- hilfreich: gute Fachliteratur als Vorbild nehmen (für Formulierungen und Notaion)
- Wenn es schwierig ist eine gute Formulierung zu finden, dann erstmal eine "schlechte" Formulierung in Rot in den Text schreiben
- Versionsverwaltung (git) für den LaTeX-Quellcode sinnvoll. Lokal oder auf Server. Nicht das gleiche Repo wie für den Programm-Quelltext verwenden. Häufige Commits sind hilfreich, wenn man später nochmal zurück möchte.


\usepackage{color}
\newcommand{\tcred}[1]{\textcolor{red}{#1}} % für tmp-Hervorhebungen

\tcred{subobtimale Formulierung}



## Inhalt


- Aufgabenstellung möglichst vollständig abgearbeitet. Wenn das nicht geht dann Begründung in der Einleitung ("nach Rücksprache mit dem Betreuer wurde der Schwerpunkt... ").
- Ggf. Hilfreich: Abschnitt "Präzisierung der Aufgabenstellung" -> konkrete Forschungsfragen aufstellen (1- 10)

- Arbeit gut und nachvollziehbar gegliedert (roten Faden)

- gut verständlicher Text (kein unbekanntes Wissen voraussetzen: Sie selbst sollten die Arbeit verstehen, auch wenn sie ein anderes Thema bearbeitet hätten)

- gute Quellenarbeit: Behauptungen, die Sie nicht selber erklären, sollten mit Literatur belegt sein. Möglichst mit Angabe von Abschnitt oder Seite, also z.B. [11, Abschnitt 5.2].

## Formalia

- Bedeutung aller Variablen einführen / erklären. Aufführung im Symbolverzeichnis reicht nicht.

- Formeln gehören zum jeweils umgebenden Satz, d.h. nach einer Formel muss dann meist auch ein Komma oder ein Punkt stehen. Am besten an Beispielen orientieren. Achtung: Manche Veröffentlichungen sind dafür kein gutes Beispiel.
- Bildunterschriften enden mit einem Punkt.
- LaTeX bietet optionale Kurzversion für das Abbildungsverzeichnis bei zu langen Bildunterschriften
- \eqref statt \ref für Verweise auf Formeln

- Eine Grafik oder Tabelle erst im Text erwähnen, bevor sie erscheint
- Verwendung von := (Definition von a durch a:= b + c) und \stackrel{!}{=} (Forderung/Bedingung) wo es angebracht ist erleichtert das Verständnis
- Subequations kann man je nach Label zusammen oder getrennt referenzieren: "Gleichung (24)" oder "Gleichung (24b)"
- Deutsche Anführungszeichen bei Verwendung mit babel: "`Hallo Welt"'

- Sätze möglichst nicht mit Symbol beginnen. Schlecht: "$m$ bezeichnet dabei die Masse." Besser: "Dabei bezeichnet $m$ die Masse."

- Grafiken: Achsen beschriften!, Symbole und Text lesbar (Schriftgröße/Kontrast), Linien gut erkennbar (Farbe/Linienstil)
- Vektorgrafiken besser als Pixelgrafiken. PNG meist besser als JPG (wegen Kompressionsverlusten bei JPG)

- Zum "Hübsch-Machen" von Grafiken müssen typischerweies plot-Parameter angepasst werden, z.B. plt.rcParams["font.size"] = 14, plt.rcParams['figure.subplot.bottom'] = .265, ....

- Grafiken müssen typischerweise 10 bis 100 mal erstellt werden bis sie "richtig schön" sind. Deshalb ist es sinnvoll,
die Daten für die Erstellung separat zu speichern (z.B. numpy.save(...) oder pickle.dump(...)) und die Erstellung der Grafik(en) in einem Separaten Skript oder Notebook durchzuführen (numpy.load(...), pickle.load(...))

- Quellen in Büchern möglichst mit konkreter Angabe, Abschnitt, Seite, etc.

- Verlängerungsantrag muss spätestens 4 Wochen vor offiziellem Abgabetermin beim Vorsitzenden des Prüfungsausschusses sein.


- Abgabe: 2 Exemplare, eines davon muss die originale Aufgabenstellung enthalten.


## Verteidigung


- Verteidigung Diplomarbeit: 30min Vortrag + Fragen
- Verteidigung Studienarbeit: 25min Vortrag + Fragen
- Richtwert: 0.5 bis 1 Folie (mit echtem Inhalt) pro Minute. -> ca. 30 Folien für den Vortrag

- Es passt nicht alles aus der Arbeit in den Vortrag. Groben überblick geben und dann ausgewählte Details vorstellen. -> Nachvollziebare Gliederung.
- Folien nicht zu voll packen damit Publikum nicht erschlagen wird. Schrittweiser Aufbau und farbliche Hervorhebungen können didaktisch sehr hilfreich sein (-> Zusatzaufwand)
- Typischerweise keine Formelnummerierungen und keine Ganzen Sätze (sondern Stichpunkte) auf Folien
- Abbildungsunterschriften eher unüblich (Ausnahme: Angabe fremder Quellen)
- Erkennbarkeit der Grafiken noch wichtiger als in der Arbeit (Kontrast auf dem Beamer ist meist schlechter (keine gelben Kurven auf weißem Grund!), Publikum sitzt ggf. weit weg von der Projektionsfläche, weniger Betrachtungszeit)
- Votrag vorher mindestens 3 mal üben. ggf. mit Diktiergerät-App aufnehmen und anhören. Der erste Übungsdurchlauf dauert erfahrungsgemäß deutlich länger. Kurze Notizen machen, welche Folien Änderungsbedarf haben.
- Vortrag spätestens ca. 3 Tage vor der Verteidigung an Betreuer schicken

- Während des Vortrags mit Zeigestab oder Laserpointer (im Sekretariat ausleihbar) zeigen. nicht mit der Hand oder dem Schatten, denn dadurch verdeckt man andere Teile der Folie.
- Empfehlenswert, die ersten und die letzten 3-5 Sätze auswendig lernen, den Rest des Vortrags frei auf Basis der eigenen Sachkenntnis halten.
- Laut, klar und deutlich sprechen.
- möglichst Blickkontakt öfter mal zum Publikum aufbauen. Schlecht: die ganze Zeit auf die Beamer-Projektion schauen, mit dem Rücken zum Publikum. Besser: seitlich hinstellen und nur wenn unbedingt nötig auf Projektionsfläche schauen. Blick auf das (entsprechend positionierte) Notebook ist besser (weil da das Gesicht zum Publikum zeigt), Blick zum Publikum ist am besten (aber auch am schwierigsten, weil man dann keine Gedächtnisstütze hat.)
