Seminar at TU Vienna: Contact Tracing, Epidemiology, Mathematical Modeling

Author: Felix Gigler

Installation:
install miniconda, then (in Anaconda Terminal):

	conda install numpy matplotlib networkx scipy ffmpeg

Note for visualizations:
beginninng with 07.01.2021 I try to name visualizations such that at some point it includes <nr of nodes>_<p> in order
to keep track of network params

Current targets:

	Mach ein paar Parametervariationen,  insbesondere für die(den) netzwerkbezogenen Parameter und schau nach, wie sich die Infektionszahlen (Zeitreihe der Summe deiner gelben/roten punkterln) verändern. Hierzu wirst du Monte Carlo simulation machen müssen - also schalt den visuellen Output aus, sonst wirst du alt dabei;). 
		    
	Implementier eine Quarantänemodell: Jeder rot-gewordene Punkt triggert sich eine gewisse Zeit nach seiner rot-werdung ein quarantäne Event das seine Kanten für eine gewisse Zeit für transmissionen disabled.
    		Unabsichtlich leicht anders (aber meiner meinung nach einigermaßen äquivalent) implementiert: Statt die kanten selbst zu deaktivieren, werden kontakt-events des individuums vorerst ausgesetzt.
	Implementier ein Tracingmodell. Jeder kontaktpartner eines rot-gewordene Punkts triggert sich eine gewisse Zeit nach dessen rot-werdung ein quarantäne Event das seine Kanten für eine gewisse Zeit für transmissionen disabled.
