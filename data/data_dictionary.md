# Data Dictionary

This document describes the structure and contents of the `steam_games_cleaned.csv` dataset containing information about 42,497 Steam games.

## Dataset Overview
- **Total Records**: 42,497 games
- **Total Columns**: 35 fields
- **Source**: Steam platform

## Column Descriptions

### Game Identification
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `app_id` | Integer | Unique Steam application identifier | 730 |
| `title` | String | Game title/name | Counter-Strike 2 |

### Release Information
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `release_date` | String | Game release date in YYYY-MM-DD format | 2012-08-21 |
| `release_year` | Float | Year of release | 2012.0 |
| `game_age_years` | Float | Age of game in years from current date | 12.36 |
| `age_bucket` | String | Categorical age grouping (<1y, 1-3y, 3-5y, 5-10y, 10y+) | 10y+ |

### Game Classification
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `genres_clean` | String | Semicolon-separated list of game genres | action;free to play |
| `main_genre` | String | Primary genre category (13 unique values) | action |
| `categories_clean` | String | Comma-separated game features/categories | cross-platform multiplayer, steam trading cards |

### Developer & Publisher
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `developer` | String | Game developer name | Valve |
| `publisher` | String | Game publisher name | Valve |

### Pricing Information
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `original_price_clean` | Float | Original listed price (currency units) | 2999.0 |
| `discount_pct_clean` | Float | Discount percentage if on sale | 50.0 |
| `discounted_price_clean` | Float | Price after discount applied | 0.0 |
| `final_price` | Float | Final price to purchase | 0.0 |
| `is_free_to_play` | Boolean | Whether game is free-to-play | True |
| `is_on_sale` | Boolean | Whether game is currently on sale | False |

### DLC & Content
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `dlc_available` | Integer | Number of DLC items available | 1 |
| `has_dlc` | Boolean | Whether game has any DLC | True |

### Age Rating
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `age_rating` | Integer | Age rating code | 0 |
| `age_rating_category` | String | Age rating category (All Ages or 7+) | All Ages |

### Platform Support
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `win_support` | Boolean | Windows platform support | True |
| `mac_support` | Boolean | macOS platform support | False |
| `linux_support` | Boolean | Linux platform support | True |
| `multi_platform` | Boolean | Whether available on multiple platforms | True |

### Awards & Access
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `awards` | Integer | Number of awards received | 1 |
| `is_early_access` | Boolean | Whether game is in early access | False |

### Review Metrics - Overall
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `overall_review` | String | Overall review sentiment (9 categories) | Very Positive |
| `overall_review_pct` | Float | Overall positive review percentage | 87.0 |
| `overall_review_count` | Float | Total number of overall reviews | 8062218.0 |
| `log_overall_review_count` | Float | Natural log of overall review count | 15.90 |

### Review Metrics - Recent
| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `recent_review` | String | Recent review sentiment (7 categories) | Mostly Positive |
| `recent_review_pct` | Float | Recent positive review percentage | 79.0 |
| `recent_review_count` | Float | Number of recent reviews | 57466.0 |
| `log_recent_review_count` | Float | Natural log of recent review count | 10.96 |
