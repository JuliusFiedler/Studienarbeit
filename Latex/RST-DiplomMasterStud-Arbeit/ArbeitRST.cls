\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\ProvidesClass{ArbeitRST}[2016/10/13 Vorlage fuer Arbeiten am RST]

\RequirePackage{kvoptions,pdftexcmds}
\SetupKeyvalOptions{family=RSTArbeit,prefix=RST@
}
\DeclareStringOption[valarbeit]{arbeit}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessKeyvalOptions*\relax

% Laden der KOMA-Script Buchklasse
\LoadClass[captions=tableheading, % Beschriftung oberhalb von Tabellen
12pt,                             % Schriftgröße
chapterprefix=true,               % Einfügen des Wortes "Kapitel" am Anfang jedes Kapitels
numbers=noendperiod,              % Keine Punkte nach den Nummerierungen
oneside,                          % Schalter zum Wechsel zwischen ein- und zweispaltigem Satz
listof=totoc,                       % Abbildungsverzeichnis in das Inhaltsverzeichnis aufnehmen
toc=
bibliography=totoc,                 % Literaturverzeichnis in das Inhaltsverzeichnis aufnehmen
headsepline=true,                 % Titelzeile in Satzspiegel einbeziehen
paper=a4,                         % Papierformat
DIV=14,                           % Satzspiegel festlegen (DIV=15 schafft noch mehr Platz)
enabledeprecatedfontcommands      % Wegen \sc in gerabbrv.bst
]{scrbook}

% Laden der KOMA-Script Klasse zur Gestaltung der Kopfzeile
\RequirePackage{scrpage2}

% Grafikpaket
\RequirePackage{graphicx}

% Codierung der Eingabedateien anpassen
\RequirePackage[utf8]{inputenc}

% Font codierung einstellen
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}

% Deutsche Bezeichnungen laden
\RequirePackage[ngerman]{babel}

% Mathematik-Pakete
\RequirePackage{amsmath,amssymb,amsfonts,amsthm,dsfont}

% Auslassungspunkte
\RequirePackage{ellipsis}

% Nummerierung der Abbildungen ändern
\RequirePackage{chngcntr}

% Schöne Tabellen
\RequirePackage{booktabs}

% Biblatex verwenden
\RequirePackage{biblatex}

% Leerzeichen der Anführungsstriche anpassen
\RequirePackage{xspace}

% Modifikationen an Matrizen etc.
\RequirePackage{array}

% Paket zur typografischen Verbesserung des Dokuments
\RequirePackage{microtype}

% Pakete für Bildunterschriften
\RequirePackage[format=plain,labelsep=endash,labelfont=bf,margin=0pt]{caption}
\RequirePackage{subcaption}
\DeclareCaptionLabelFormat{myparens}{#2)}
\captionsetup[subfigure]{format=hang,labelfont=normalfont,labelformat=myparens,labelsep=space,margin=15pt}

% Verweise etc.
\RequirePackage{hyperref}

% Anpassen der Texte auf die jeweilige Form der Arbeit
\newif\ifzeigegrad
\zeigegradfalse

% Fußnoten ohne die nervigen Einrückungen bei Zeilenwechsel
\deffootnote[1em]{1em}{1em}{\textsuperscript{\thefootnotemark}}


\newcommand{\arbeitartikel}{die}
\newcommand{\arbeitwoeingereicht}{dem Prüfungsausschuss}
\newcommand{\arbeiteingereicht}{eingereichte}

\ifnum\pdf@strcmp{\RST@arbeit}{forsch}=0
	\subject{Bericht zum Forschungspraktikum}
	\renewcommand{\arbeitartikel}{den}
	\renewcommand{\arbeitwoeingereicht}{an}
	\renewcommand{\arbeiteingereicht}{eingereichten}
\else\ifnum\pdf@strcmp{\RST@arbeit}{master}=0
	\subject{Masterarbeit}
	\zeigegradtrue
\else\ifnum\pdf@strcmp{\RST@arbeit}{studie}=0
	\subject{Studienarbeit}
	\renewcommand{\arbeitwoeingereicht}{an}
\else
	\subject{Diplomarbeit}
	\zeigegradtrue
\fi\fi\fi%


% Seitenstil festlegen
\pagestyle{scrheadings}

% Kopfzeile unterstreichen
\KOMAoption{headsepline}{true}
\automark[section]{chapter}
\ihead{\headmark}
\chead{}
\ohead{\pagemark}
\ifoot{}
\cfoot[\pagemark]{}
\ofoot{}

\setheadsepline[text]{.5pt}

% Schriftart der Überschriften ändern
\setkomafont{sectioning}{\normalfont\bfseries}

% Aufzählungstil anpassen
\renewcommand{\labelitemi}{--}

% Bildunterschrift anpassen
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}

% Anführungsstriche anpassen
\expandafter\def\expandafter\grqq\expandafter{\grqq\xspace}

% Eigene Makros
\DeclareMathOperator{\diag}{\mathrm{diag}}
\newcommand{\partiell}[3][]{\frac{\partial^{#1}#2}{\partial{#3}^{#1}}}
\newcommand{\diff}[3][]{\frac{\mathrm{d}^{#1}#2}{\mathrm{d}{#3}^{#1}}}
\newcommand{\dr}{\mathrm{d}}
\newcommand{\vect}[1]{\boldsymbol #1}
\newcommand{\Reals}{\mathds R}
\newcommand{\Compl}{\mathds C}
\DeclareMathOperator{\Real}{\mathfrak R}
\DeclareMathOperator{\Imag}{\mathfrak I}
\newcommand{\norm}[1]{\left\|#1\right\|}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\skalprod}[2]{\langle#1,#2\rangle}
\DeclareMathOperator{\grad}{\mathrm{grad}}
\renewcommand{\div}{\mathrm{div}}

% Umgebungen
\newtheorem{definition}{Definition}[chapter]
\newtheorem{beispiel}{Beispiel}[chapter]
\newtheorem{lemma}{Lemma}[chapter]
\newtheorem{bemerkung}{Bemerkung}[chapter]

% Titelseite
\makeatletter
\newif\ifdeftitel
\deftitelfalse
\newcommand*{\@betreuer}{}%
\newcommand*{\@geburtsort}{Geburtsort}%
\newcommand*{\@geburtsdatum}{Geburtsdatum}%
\newcommand{\betreuer}[1]{\g@addto@macro{\@betreuer}{& #1 \\}}
\newcommand{\geburtsort}[1]{\renewcommand*{\@geburtsort}{#1}}%
\newcommand{\geburtsdatum}[1]{\renewcommand*{\@geburtsdatum}{#1}}%
\renewcommand*\maketitle{%
\deftiteltrue
\begin{titlepage}
\begin{center}
{\LARGE \textbf{Technische Universität Dresden}}\\[2ex]
{\large Fakultät Elektrotechnik und Informationstechnik}\\[2ex]
{\large Institut für Regelungs- und Steuerungstheorie}\\

\vfill
{\usekomafont{subject} \@subject}\\[4ex]
{\usekomafont{title}\Large \@title}\\[4ex]
\ifx\@subtitle\empty\else{\usekomafont{subtitle} \@subtitle}\\\fi

\vfill

\begin{tabular}{rl}

vorgelegt von: & \@author\\

geboren am: & \@geburtsdatum{} in \@geburtsort{}
\end{tabular}


\ifzeigegrad
\vfill
zum Erlangen des akademischen Grades\\[1ex]
\ifnum\pdf@strcmp{\RST@arbeit}{master}=0
{\Large\textbf{Master of Science}}\\
(M.\ Sc.)\\
\else
{\Large\textbf{Diplomingenieur}}\\
(Dipl.-Ing.)\\
\fi\fi%

\vfill

\begin{tabular}{ll}
\ifx\@betreuer\empty\else Betreuer: \@betreuer\fi
Verantwortlicher Hochschullehrer: & Prof. Dr.-Ing. habil. Dipl.-Math. K. Röbenack\\
Tag der Einreichung: & \@date
\end{tabular}
\end{center}
\end{titlepage}
\newpage
\thispagestyle{empty}
\null\vfill
\begin{center}
	Bitte ersetzen Sie diese Seite vor dem Binden mit der Aufgabenstellung.
\end{center}
\vfill
}%

 %Selbstständigkeitserklärung
\newif\ifdefselbst
\defselbstfalse
\newcommand*{\@selbstort}{Dresden}%
\newcommand*{\@selbstdatum}{\@date}%
\newcommand{\selbstort}[1]{\renewcommand{\@selbstort}{#1}}%
\newcommand{\selbstdatum}[1]{\renewcommand{\@selbstdatum}{#1}}%
\newcommand{\selbststaendigkeitserklaerung}{
\defselbsttrue
\clearpage
\thispagestyle{empty}
\vspace*{4cm}
\noindent\textbf{\Large Selbstständigkeitserklärung}\\[4ex]
Hiermit erkläre ich, dass ich \arbeitartikel{} von mir am heutigen Tage \arbeitwoeingereicht{} der Fakultät Elektrotechnik und Informationstechnik \arbeiteingereicht{} \@subject{} zum Thema
\begin{center}
	{\usekomafont{subtitle}\normalsize\@title}
\end{center}
selbstständig und ohne Benutzung anderer als der angegebenen Hilfsmittel angefertigt
habe. Alle Stellen, die wörtlich oder sinngemäß aus veröffentlichten oder nicht
veröffentlichten Schriften entnommen sind, wurden als solche kenntlich gemacht.

\vspace*{2cm}

\noindent\@selbstort, \@selbstdatum\hfill\@author
}
% Kurzfassung
% 1. Argument - deutsch
% 2. Argument - englisch
\newif\ifdefkurz
\defkurzfalse
\newcommand{\kurzfassung}[2]{
\defkurztrue
\clearpage
\thispagestyle{empty}
\vspace*{4cm}
\noindent\textbf{\Large Kurzfassung}\\[2ex]
#1\\
\vspace*{2cm}\\
\noindent\textbf{\Large Abstract}\\[2ex]
#2
}

% Seitenzahl im Inhaltsverzeichnis entfernen
%\g@addto@macro{\tableofcontents}{\thispagestyle{empty}}

\newcounter{seiteeins}
\setcounter{seiteeins}{0}
\AtBeginDocument{\setcounter{page}{\value{seiteeins}}}
\AtEndDocument{
\setcounter{seiteeins}{0}
\ifdefselbst
  \addtocounter{seiteeins}{-1}
\fi
\ifdeftitel
  \addtocounter{seiteeins}{-2}
\fi
\ifdefkurz
	\addtocounter{seiteeins}{-1}
\fi
\protected@write\@auxout{}{\string\setcounter{seiteeins}{\theseiteeins}}
}
\makeatother

\endinput