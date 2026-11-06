
from pathlib import Path
import html
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Andrade Fantasy Football | League History",
    page_icon="🏈",
    layout="wide",
)

DATA_FILE = Path("data/matchups.parquet")
ALIASES_FILE = Path("data/manager_aliases.csv")

REQUIRED_COLUMNS = [
    "Year",
    "Game Type",
    "Week",
    "Manager 1",
    "Team 1",
    "Team 1 Score",
    "Team 2 Score",
    "Team 2",
    "Manager 2",
]

# Optional: map alternative manager names to one identity.
# Example:
# Alias,Manager
# Thomas E,Thomas Eliot
# Tom Eliot,Thomas Eliot


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(file_path):
    df = pd.read_parquet(file_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")

    df["Team 1 Score"] = pd.to_numeric(
        df["Team 1 Score"], errors="coerce"
    )

    df["Team 2 Score"] = pd.to_numeric(
        df["Team 2 Score"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Year",
            "Week",
            "Manager 1",
            "Manager 2",
            "Team 1 Score",
            "Team 2 Score",
        ]
    )

    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)

    for col in [
        "Manager 1",
        "Manager 2",
        "Team 1",
        "Team 2",
        "Game Type",
    ]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[
        (df["Manager 1"] != "")
        & (df["Manager 2"] != "")
        ]

    # Apply optional manager identity mappings.
    if ALIASES_FILE.exists():
        aliases = pd.read_csv(ALIASES_FILE)

        if {"Alias", "Manager"}.issubset(aliases.columns):
            mapping = dict(
                zip(
                    aliases["Alias"].astype(str).str.strip(),
                    aliases["Manager"].astype(str).str.strip(),
                )
            )

            df["Manager 1"] = df["Manager 1"].replace(mapping)
            df["Manager 2"] = df["Manager 2"].replace(mapping)

    # The score is the source of truth for matchup results.
    df["Margin"] = (
            df["Team 1 Score"] - df["Team 2 Score"]
    )

    df["Manager Winner"] = None

    df.loc[
        df["Margin"] > 0, "Manager Winner"
    ] = df.loc[df["Margin"] > 0, "Manager 1"]

    df.loc[
        df["Margin"] < 0, "Manager Winner"
    ] = df.loc[df["Margin"] < 0, "Manager 2"]

    df["Is Tie"] = df["Margin"].abs() < 0.001

    return df.reset_index(drop=True)


def create_manager_games(df):
    """
    Convert each matchup into two manager-centric records.

    Every game produces:
      - One record for Manager 1
      - One record for Manager 2

    This makes manager-level analytics much easier.
    """

    first = pd.DataFrame({
        "Year": df["Year"],
        "Week": df["Week"],
        "Game Type": df["Game Type"],
        "Manager": df["Manager 1"],
        "Team": df["Team 1"],
        "Opponent": df["Manager 2"],
        "Opponent Team": df["Team 2"],
        "Points For": df["Team 1 Score"],
        "Points Against": df["Team 2 Score"],
    })

    second = pd.DataFrame({
        "Year": df["Year"],
        "Week": df["Week"],
        "Game Type": df["Game Type"],
        "Manager": df["Manager 2"],
        "Team": df["Team 2"],
        "Opponent": df["Manager 1"],
        "Opponent Team": df["Team 1"],
        "Points For": df["Team 2 Score"],
        "Points Against": df["Team 1 Score"],
    })

    games = pd.concat(
        [first, second],
        ignore_index=True
    )

    games["Margin"] = (
            games["Points For"] - games["Points Against"]
    )

    games["Win"] = games["Margin"] > 0.001
    games["Loss"] = games["Margin"] < -0.001
    games["Tie"] = games["Margin"].abs() <= 0.001

    games["Result"] = "Tie"
    games.loc[games["Win"], "Result"] = "W"
    games.loc[games["Loss"], "Result"] = "L"

    return games.sort_values(
        ["Year", "Week", "Game Type"]
    ).reset_index(drop=True)


def calculate_summary(games):
    summary = (
        games.groupby("Manager")
        .agg(
            Games=("Manager", "size"),
            Wins=("Win", "sum"),
            Losses=("Loss", "sum"),
            Ties=("Tie", "sum"),
            Points_For=("Points For", "sum"),
            Points_Against=("Points Against", "sum"),
            Avg_Points=("Points For", "mean"),
            Highest_Score=("Points For", "max"),
            Lowest_Score=("Points For", "min"),
        )
        .reset_index()
    )

    summary["Win %"] = (
            summary["Wins"] / summary["Games"]
    )

    summary["Point Differential"] = (
            summary["Points_For"] - summary["Points_Against"]
    )

    return summary.sort_values(
        ["Wins", "Points_For"],
        ascending=False
    )


def calculate_head_to_head(games, manager):
    """
    Calculate one manager's record against every opponent.
    """

    subset = games[games["Manager"] == manager]

    if subset.empty:
        return pd.DataFrame()

    h2h = (
        subset.groupby("Opponent")
        .agg(
            Games=("Opponent", "size"),
            Wins=("Win", "sum"),
            Losses=("Loss", "sum"),
            Ties=("Tie", "sum"),
            Points_For=("Points For", "sum"),
            Points_Against=("Points Against", "sum"),
            Avg_Points=("Points For", "mean"),
        )
        .reset_index()
    )

    h2h["Win %"] = h2h["Wins"] / h2h["Games"]

    h2h["Point Differential"] = (
            h2h["Points_For"] - h2h["Points_Against"]
    )

    return h2h.sort_values(
        ["Games", "Wins"],
        ascending=[False, False]
    )


def calculate_streaks(games):
    """
    Calculate the longest consecutive winning and losing streaks.
    Games are ordered chronologically. Ties break a streak.
    """
    results = {}

    for manager, group in games.groupby("Manager"):
        group = group.sort_values(
            ["Year", "Week"]
        )

        current_win = 0
        current_loss = 0
        max_win = 0
        max_loss = 0

        for result in group["Result"]:
            if result == "W":
                current_win += 1
                current_loss = 0

            elif result == "L":
                current_loss += 1
                current_win = 0

            else:
                current_win = 0
                current_loss = 0

            max_win = max(max_win, current_win)
            max_loss = max(max_loss, current_loss)

        results[manager] = {
            "Longest Win Streak": max_win,
            "Longest Losing Streak": max_loss,
        }

    return results


def calculate_manager_profile(games, manager):
    subset = games[games["Manager"] == manager]

    if subset.empty:
        return {}

    wins = int(subset["Win"].sum())
    losses = int(subset["Loss"].sum())
    ties = int(subset["Tie"].sum())
    total = len(subset)

    h2h = calculate_head_to_head(games, manager)

    never_beaten = h2h[
        (h2h["Wins"] == 0)
        & (h2h["Games"] > 0)
        ]["Opponent"].tolist()

    never_lost_to = h2h[
        (h2h["Losses"] == 0)
        & (h2h["Games"] > 0)
        ]["Opponent"].tolist()

    # Most frequently played opponent.
    biggest_rival = (
        h2h.sort_values(
            ["Games", "Losses"],
            ascending=[False, False]
        ).iloc[0]["Opponent"]
        if not h2h.empty else "N/A"
    )

    # Highest percentage of losses against an opponent.
    # Require at least two matchups to avoid tiny samples.
    qualified = h2h[h2h["Games"] >= 2].copy()

    biggest_nemesis = "N/A"

    if not qualified.empty:
        qualified["Loss %"] = (
                qualified["Losses"] / qualified["Games"]
        )

        biggest_nemesis = qualified.sort_values(
            ["Loss %", "Games"],
            ascending=[False, False]
        ).iloc[0]["Opponent"]

    # Highest win percentage against an opponent.
    biggest_victim = "N/A"

    if not qualified.empty:
        biggest_victim = qualified.sort_values(
            ["Win %", "Games"],
            ascending=[False, False]
        ).iloc[0]["Opponent"]

    streaks = calculate_streaks(games).get(manager, {})

    return {
        "Games": total,
        "Wins": wins,
        "Losses": losses,
        "Ties": ties,
        "Win %": wins / total if total else 0,
        "Points For": subset["Points For"].sum(),
        "Points Against": subset["Points Against"].sum(),
        "Average Points": subset["Points For"].mean(),
        "Highest Score": subset["Points For"].max(),
        "Lowest Score": subset["Points For"].min(),
        "Biggest Rival": biggest_rival,
        "Biggest Nemesis": biggest_nemesis,
        "Biggest Victim": biggest_victim,
        "Never Beaten": never_beaten,
        "Never Lost To": never_lost_to,
        **streaks,
    }

def calculate_yearly_stats(games):
    yearly = (
        games.groupby(["Year", "Manager"])
        .agg(
            Games=("Manager", "size"),
            Wins=("Win", "sum"),
            Losses=("Loss", "sum"),
            Ties=("Tie", "sum"),
            Points=("Points For", "sum"),
            Avg_Points=("Points For", "mean"),
        )
        .reset_index()
    )

    yearly["Win %"] = (
            yearly["Wins"] / yearly["Games"]
    )

    return yearly


def calculate_records(games):
    if games.empty:
        return {}

    return {
        "Most Points in a Game": games["Points For"].max(),
        "Lowest Points in a Game": games["Points For"].min(),
        "Largest Victory": games["Margin"].max(),
        "Worst Defeat": games["Margin"].min(),
        "Highest Scoring Game": (
                games["Points For"] + games["Points Against"]
        ).max(),
    }

def generate_email_report(games, summary):
    """
    Generate a standalone HTML league report.
    Download it and attach it to an email.
    """

    total_games = len(games) // 2

    leader = (
        summary.iloc[0]["Manager"]
        if not summary.empty else "N/A"
    )

    standings = summary[
        [
            "Manager",
            "Wins",
            "Losses",
            "Ties",
            "Win %",
            "Points_For",
            "Point Differential",
        ]
    ].copy()

    standings["Win %"] = (
                                 standings["Win %"] * 100
                         ).round(1).astype(str) + "%"

    standings["Points_For"] = (
        standings["Points_For"].round(2)
    )

    standings["Point Differential"] = (
        standings["Point Differential"].round(2)
    )

    table_html = standings.to_html(
        index=False,
        border=0,
        classes="standings"
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #101827;
            color: #f5f5f5;
            padding: 30px;
        }}

        .container {{
            max-width: 900px;
            margin: auto;
        }}

        h1 {{ color: #ffffff; }}

        .card {{
            background: #1e293b;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}

        th {{
            color: #38bdf8;
        }}

        .highlight {{
            color: #38bdf8;
            font-size: 24px;
            font-weight: bold;
        }}

        .note {{
            color: #94a3b8;
            font-size: 12px;
        }}
    </style>
    </head>

    <body>
    <div class="container">

        <h1>🏈 Fantasy Football League History</h1>

        <div class="card">
            <p>All-Time Matchups</p>
            <div class="highlight">{total_games}</div>
        </div>

        <div class="card">
            <p>All-Time Wins Leader</p>
            <div class="highlight">
                {html.escape(str(leader))}
            </div>
        </div>

        <h2>All-Time Standings</h2>

        {table_html}

        <p class="note">
            Generated from historical fantasy football matchups.
            Win percentage excludes no games, including ties in its denominator.
        </p>

    </div>
    </body>
    </html>
    """


# ============================================================
# UI HELPERS
# ============================================================

def metric(label, value):
    st.metric(label, value)


def format_points(value):
    return f"{value:,.2f}"


def display_manager_profile(games, manager):
    profile = calculate_manager_profile(games, manager)

    if not profile:
        st.warning("No matchups available.")
        return

    st.title(f"🏈 {manager}")

    # Career overview
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Wins", profile["Wins"])
    c2.metric("Losses", profile["Losses"])
    c3.metric("Win Percentage", f'{profile["Win %"]:.1%}')
    c4.metric("Games Played", profile["Games"])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Career Points",
        format_points(profile["Points For"])
    )

    c2.metric(
        "Average Points",
        format_points(profile["Average Points"])
    )

    c3.metric(
        "Point Differential",
        format_points(
            profile["Points For"] - profile["Points Against"]
        )
    )

    c4.metric(
        "Highest Score",
        format_points(profile["Highest Score"])
    )

    st.divider()

    # Rivalries
    st.subheader("⚔️ Rivalries & Head-to-Head")

    c1, c2, c3 = st.columns(3)

    c1.metric("Biggest Rival", profile["Biggest Rival"])
    c2.metric("Biggest Nemesis", profile["Biggest Nemesis"])
    c3.metric("Biggest Victim", profile["Biggest Victim"])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Managers this player has NEVER beaten**")

        if profile["Never Beaten"]:
            for opponent in profile["Never Beaten"]:
                st.write(f"❌ {opponent}")
        else:
            st.success("No winless head-to-head records.")

    with c2:
        st.markdown("**Managers this player has NEVER lost to**")

        if profile["Never Lost To"]:
            for opponent in profile["Never Lost To"]:
                st.write(f"🏆 {opponent}")
        else:
            st.write("No undefeated head-to-head records.")

    st.divider()

    # Streaks
    st.subheader("🔥 Career Streaks")

    c1, c2 = st.columns(2)

    c1.metric(
        "Longest Winning Streak",
        profile["Longest Win Streak"]
    )

    c2.metric(
        "Longest Losing Streak",
        profile["Longest Losing Streak"]
    )

    st.divider()

    # Head-to-head table
    st.subheader("Head-to-Head Records")

    h2h = calculate_head_to_head(games, manager)

    if not h2h.empty:
        display = h2h[
            [
                "Opponent",
                "Games",
                "Wins",
                "Losses",
                "Ties",
                "Win %",
                "Points_For",
                "Points_Against",
            ]
        ].copy()

        display["Win %"] = (
                                   display["Win %"] * 100
                           ).round(1).astype(str) + "%"

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Historical performance
    st.subheader("📈 Season-by-Season Performance")

    yearly = (
        games[games["Manager"] == manager]
        .groupby("Year")
        .agg(
            Wins=("Win", "sum"),
            Losses=("Loss", "sum"),
            Points=("Points For", "sum"),
            Average=("Points For", "mean"),
        )
        .reset_index()
    )

    if not yearly.empty:
        fig = px.bar(
            yearly,
            x="Year",
            y="Points",
            title="Total Points by Season",
            text_auto=".0f",
        )

        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            yearly,
            x="Year",
            y="Average",
            markers=True,
            title="Average Points Per Matchup",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            yearly,
            use_container_width=True,
            hide_index=True
        )

    # Fantasy team name history
    st.subheader("Historical Fantasy Team Names")

    teams = (
        games[games["Manager"] == manager]
        .groupby(["Year", "Team"])
        .size()
        .reset_index(name="Games")
        .sort_values("Year")
    )

    st.dataframe(
        teams,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.sidebar.title("🏈 League Analytics")

    if not DATA_FILE.exists():
        st.error(
            f"Cannot find {DATA_FILE}. "
            "Place your Parquet file in the data directory."
        )
        st.stop()

    try:
        raw = load_data(str(DATA_FILE))
    except Exception as exc:
        st.error(f"Unable to load data: {exc}")
        st.stop()

    manager_games = create_manager_games(raw)

    managers = sorted(
        manager_games["Manager"].unique().tolist()
    )

    years = sorted(
        manager_games["Year"].unique().tolist()
    )

    if not managers:
        st.warning("No manager records found.")
        st.stop()

    # --------------------------------------------------------
    # GLOBAL FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("Filters")

    selected_year = st.sidebar.selectbox(
        "Season",
        ["All Seasons"] + years
    )

    game_types = sorted(
        manager_games["Game Type"].unique().tolist()
    )

    selected_type = st.sidebar.selectbox(
        "Game Type",
        ["All Game Types"] + game_types
    )

    filtered = manager_games.copy()

    if selected_year != "All Seasons":
        filtered = filtered[
            filtered["Year"] == selected_year
            ]

    if selected_type != "All Game Types":
        filtered = filtered[
            filtered["Game Type"] == selected_type
            ]

    if filtered.empty:
        st.warning("No games match the selected filters.")
        st.stop()

    # --------------------------------------------------------
    # DASHBOARD NAVIGATION
    # --------------------------------------------------------

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Dashboard",
        [
            "League Overview",
            "Manager Profiles",
            "Historical Records",
            "Email / Export",
        ]
    )

    st.sidebar.divider()

    st.sidebar.caption(
        f"{len(managers)} managers | "
        f"{len(filtered) // 2} matchups displayed"
    )

    # --------------------------------------------------------
    # LEAGUE OVERVIEW
    # --------------------------------------------------------

    if page == "League Overview":

        st.title("🏈 Fantasy Football League History")
        st.caption("All-time league performance and historical analytics")

        summary = calculate_summary(filtered)

        total_games = len(filtered) // 2

        total_points = filtered["Points For"].sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Managers", len(managers))
        c2.metric("Matchups", total_games)
        c3.metric("Total Points", format_points(total_points))
        c4.metric(
            "Average Points / Game",
            format_points(
                filtered["Points For"].mean()
            )
        )

        st.divider()

        st.subheader("All-Time Manager Rankings")

        rankings = summary.copy()

        rankings["Win %"] = (
                                    rankings["Win %"] * 100
                            ).round(1).astype(str) + "%"

        st.dataframe(
            rankings,
            use_container_width=True,
            hide_index=True
        )

        # Career wins
        st.subheader("🏆 Career Wins")

        fig = px.bar(
            summary.sort_values("Wins"),
            x="Wins",
            y="Manager",
            orientation="h",
            text="Wins",
            title="All-Time Wins",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Career points
        st.subheader("💰 Career Points Scored")

        fig = px.bar(
            summary.sort_values("Points_For"),
            x="Points_For",
            y="Manager",
            orientation="h",
            title="All-Time Points Scored",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Annual performance
        st.subheader("📅 Annual Performance")

        yearly = calculate_yearly_stats(filtered)

        if not yearly.empty:
            fig = px.line(
                yearly,
                x="Year",
                y="Points",
                color="Manager",
                markers=True,
                title="Points Scored by Manager Each Year",
            )

            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                yearly,
                x="Year",
                y="Wins",
                color="Manager",
                barmode="group",
                title="Wins by Manager Each Year",
            )

            st.plotly_chart(fig, use_container_width=True)

        # Cumulative career points
        st.subheader("📈 Cumulative Career Points")

        cumulative = yearly.sort_values(
            ["Manager", "Year"]
        ).copy()

        cumulative["Cumulative Points"] = (
            cumulative.groupby("Manager")["Points"].cumsum()
        )

        fig = px.line(
            cumulative,
            x="Year",
            y="Cumulative Points",
            color="Manager",
            markers=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Annual standings
        st.subheader("Season Standings")

        if not yearly.empty:
            season = st.selectbox(
                "Select season for standings",
                sorted(yearly["Year"].unique(), reverse=True)
            )

            season_standings = yearly[
                yearly["Year"] == season
                ].sort_values(
                ["Wins", "Points"],
                ascending=False
            )

            st.dataframe(
                season_standings,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # MANAGER PROFILES
    # --------------------------------------------------------

    elif page == "Manager Profiles":

        st.title("👤 Manager Profiles")

        selected_manager = st.selectbox(
            "Select Manager",
            managers
        )

        display_manager_profile(
            filtered,
            selected_manager
        )

    # --------------------------------------------------------
    # HISTORICAL RECORDS
    # --------------------------------------------------------

    elif page == "Historical Records":

        st.title("🏆 League Historical Records")

        records = calculate_records(filtered)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Highest Individual Score",
            format_points(records["Most Points in a Game"])
        )

        c2.metric(
            "Lowest Individual Score",
            format_points(records["Lowest Points in a Game"])
        )

        c3.metric(
            "Highest Scoring Matchup",
            format_points(records["Highest Scoring Game"])
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Largest Victory Margin",
            format_points(records["Largest Victory"])
        )

        c2.metric(
            "Worst Defeat Margin",
            format_points(abs(records["Worst Defeat"]))
        )

        st.divider()

        summary = calculate_summary(filtered)

        streaks = calculate_streaks(filtered)

        summary["Longest Win Streak"] = summary[
            "Manager"
        ].map(
            lambda x: streaks[x]["Longest Win Streak"]
        )

        summary["Longest Losing Streak"] = summary[
            "Manager"
        ].map(
            lambda x: streaks[x]["Longest Losing Streak"]
        )

        st.subheader("Winning Streak Rankings")

        st.dataframe(
            summary[
                [
                    "Manager",
                    "Longest Win Streak",
                    "Longest Losing Streak",
                ]
            ].sort_values(
                "Longest Win Streak",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Biggest League Bully")

        bully = summary.sort_values(
            "Wins",
            ascending=False
        ).iloc[0]

        st.success(
            f'{bully["Manager"]} leads the league with '
            f'{int(bully["Wins"])} career wins.'
        )

        st.subheader("Most Points Scored")

        scorer = summary.sort_values(
            "Points_For",
            ascending=False
        ).iloc[0]

        st.info(
            f'{scorer["Manager"]} has scored '
            f'{scorer["Points_For"]:,.2f} career points.'
        )

    # --------------------------------------------------------
    # EMAIL / EXPORT
    # --------------------------------------------------------

    elif page == "Email / Export":

        st.title("📧 League Reports")

        st.write(
            "Generate a report to share with the rest of the league."
        )

        summary = calculate_summary(filtered)

        report_html = generate_email_report(
            filtered,
            summary
        )

        st.download_button(
            label="Download HTML League Report",
            data=report_html,
            file_name="fantasy_league_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.caption(
            "The HTML file can be attached to an email. "
            "Recipients can open it in their browser."
        )

        st.divider()

        st.subheader("Export Raw Statistics")

        st.download_button(
            "Download Manager Rankings (CSV)",
            summary.to_csv(index=False),
            file_name="manager_rankings.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download All Matchups (CSV)",
            filtered.to_csv(index=False),
            file_name="all_matchups.csv",
            mime="text/csv",
        )

        yearly = calculate_yearly_stats(filtered)

        st.download_button(
            "Download Annual Statistics (CSV)",
            yearly.to_csv(index=False),
            file_name="annual_statistics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
