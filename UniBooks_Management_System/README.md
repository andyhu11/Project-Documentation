# UniBooks Management System

> **A comprehensive database solution for efficient bookstore inventory, sales, and personnel management.**

## 📖 Overview

**UniBooks** is a robust database management system built with **Microsoft Access**, designed to streamline the complex daily operations of a bookstore. By moving away from manual tracking, UniBooks ensures data integrity and operational efficiency.

The system manages the complete lifecycle of bookstore data—from supplier purchase orders to customer sales transactions—while providing management with actionable insights through dynamic reporting and trend analysis.

---

## ✨ Key Features

### 🛠 Core Operations

* **Secure Access Control:** Role-based login system ensures that sensitive data is protected and only accessible to authorized personnel.
* **Transaction Management:** Dedicated interfaces for **Purchase Orders** (acquisitions) and **Sell Orders** (customer transactions).
* **Automated Calculations:** Built-in engines automatically handle transaction totals and basic operational logic, reducing manual error.

### 📊 Business Intelligence

* **Dynamic Queries:**
* **Best-Sellers:** Instantly identify the most popular books to optimize stock.
* **Revenue Tracking:** Monitor monthly financial performance.


* **Strategic Reporting:**
* **Inventory Health:** Real-time reports (`BookCurrentStockAndStockLevel`) to prevent stockouts and overstocking.
* **Staff Performance:** Employee ranking reports (`StaffSales TurnoverRank`) to track sales efficiency and incentivize performance.


* **Visual Analytics:** Integrated charts for analyzing long-term sales trends.

### 🔍 Utility

* **Quick Search:** Advanced search functionality to rapidly locate book details and inventory status.

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

* **Microsoft Access** (2016 or newer recommended).
* **Windows OS** (Required for full VBA and Macro compatibility).

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/Project-Documentation.git

```


2. Navigate to the `UniBooks_Management_System` directory.
3. Locate the file **`UniBooksManagementSystem.accdb`**.

### Usage Guide

1. **Launch:** Double-click the `.accdb` file to start the application.
2. **Authentication:** The system will launch a **Login Screen**. Please enter your credentials to proceed.
3. **Navigation:** Use the main Switchboard to access the three core modules:
* **Forms:** For data entry (Orders, Customers, Books).
* **Reports:** For generating printable summaries.
* **Charts:** For visual data analysis.



> **Note:** For detailed operating procedures and technical specifications, please refer to the **[Operating Manual](https://www.google.com/search?q=./UniBooksManagementSystem_Operating%2520Manual.pdf)** included in this repository.

---

## 🚧 Roadmap & Future Enhancements

The following features are currently in the development pipeline to further enhance system automation and data integrity:

* **Automated Discount Logic:**
* Upgrade the Purchase Order module to automatically apply tiered discounts based on purchaser identity (Student vs. Publisher) and order volume.


* **Strict Inventory Validation:**
* Implement VBA-based constraints in the Sell Order module to prevent transactions that exceed available stock, ensuring absolute inventory accuracy.



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
