# UniBooks Management System

> **A comprehensive relational database solution engineered for high-integrity bookstore inventory, sales, and personnel management.**

## 📖 Overview

**UniBooks** is a robust Database Management System (DBMS) developed to digitize and optimize the complex operational workflows of modern bookstores. Addressing the inefficiencies of manual tracking, this solution leverages **Microsoft Access** and **VBA** to ensure atomic data consistency and operational scalability.

The system encompasses the full data lifecycle—facilitating secure **Purchase Order** acquisitions and **Sell Order** transactions—while empowering stakeholders with actionable business intelligence through dynamic SQL-driven reporting and trend analysis.

---

## ✨ Key Features

### 🔐 Architecture & Security

* **Role-Based Access Control (RBAC):**
* Secure login interface restricting system access based on user authorization levels.
* Ensures data integrity by preventing unauthorized modifications to sensitive inventory records.


* **Relational Design:**
* Normalized database schema designed to minimize redundancy and maintain referential integrity across Customer, Book, and Supplier entities.



### 🛠 Core Transaction Engines

* **Automated Transaction Processing:**
* **Purchase Orders:** Streamlined acquisition modules for managing supplier interactions.
* **Sell Orders:** Customer-facing interface with built-in calculation engines for immediate total computation.


* **VBA Automation:**
* Event-driven procedures handle operational logic behind the scenes, significantly reducing manual calculation errors.



### 📊 Analytics & Reporting Suite

* **Dynamic SQL Queries:**
* **Best-Seller Identification:** Aggregation algorithms to isolate high-velocity stock.
* **Revenue Analytics:** Temporal queries to track financial performance on a monthly basis.


* **Strategic Visualization:**
* **Inventory Health Reports:** Real-time monitoring (`BookCurrentStockAndStockLevel`) to mitigate stockouts and overstock risks.
* **Performance Metrics:** Staff ranking systems (`StaffSales TurnoverRank`) to quantify employee efficiency.
* **Trend Analysis:** Integrated charting tools for visualizing long-term sales trajectories.



---

## 📂 Project Structure

```text
UniBooks_Management_System/
├── UniBooksManagementSystem.accdb              # Main Database Application
├── UniBooksManagementSystem_Operating Manual.pdf   # Technical Documentation & User Guide
└── README.md                                   # Project Documentation

```

---

## 🚀 Getting Started

### Prerequisites

* **Microsoft Access** (2016 or newer required).
* **Windows OS** (Mandatory for full VBA/Macro execution support).

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/UniBooks-System.git
cd UniBooks_Management_System

```


2. **Locate the binary:**
Ensure `UniBooksManagementSystem.accdb` is present in the root directory.

### Usage Guide

1. **Initialization:**
Double-click `UniBooksManagementSystem.accdb` to mount the database. *Note: You may need to "Enable Content" to allow VBA scripts to run.*
2. **Authentication:**
The system will initialize with a **Login Screen**. Enter valid credentials to access the Switchboard.
3. **Operation Modules:**
* **Forms:** Navigate to specific modules for data entry (Orders, Customers, Inventory).
* **Reports/Charts:** Access the BI dashboard for printable summaries and visual analysis.



> **Documentation:** For deep-dive technical specifications and user workflows, refer to the **[Operating Manual](./UniBooksManagementSystem_Operating%20Manual.pdf)** included in this repository.

---

## 🚧 Roadmap & Future Enhancements

* **Logic Upgrade:**
* **Dynamic Discounting:** Implementation of conditional logic in the Purchase Order module to auto-apply tiered discounts based on identity (Student/Publisher) and volume.


* **Integrity Constraints:**
* **Stock Validation:** Developing VBA-based pre-commit triggers to strictly prevent Sell Orders that exceed current inventory levels (`Inventory < Order_Qty`), ensuring absolute stock accuracy.


* **UX Refinement:**
* Migration to a fully modal interface for focused user interaction.



---

## 🤝 Contributing

Contributions are welcome. If you would like to help implement the features listed in the Roadmap:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewFeature`).
3. Commit your Changes (`git commit -m 'Add some NewFeature'`).
4. Push to the Branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
