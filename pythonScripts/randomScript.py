import osmnx as ox
import matplotlib.pyplot as plt

# Funktion zur Abruf der OSM-Daten für eine bestimmte Straße in Hamburg
def plot_street_in_hamburg(street_name):
    # Abrufen der OSM-Daten für Hamburg
    place = 'Hamburg, Germany'
    graph = ox.graph_from_place(place, network_type='drive')

    # Abrufen der Knoten und Kanten des Graphen
    nodes, edges = ox.graph_to_gdfs(graph)

    # Filtern der Kanten nach dem Straßennamen
    street_edges = edges[edges['name'] == street_name]

    # Plotten der Karte von Hamburg
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, edge_color='gray', edge_linewidth=0.5)

    # Markieren der Straße "Jungfernstieg"
    if not street_edges.empty:
        street_edges.plot(ax=ax, color='red', linewidth=2)
    else:
        print(f"Die Straße '{street_name}' wurde nicht gefunden.")

    plt.show()

# Aufrufen der Funktion
plot_street_in_hamburg('Jungfernstieg')
